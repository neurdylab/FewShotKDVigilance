import mne
import scipy.io
from scipy import signal
import os
import glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import torch

EEG_BASE_DIR = ""
EEG_MAT_DIR = ""
FMRI_BASE_DIR = ""
EEG_INDEX_BASE_DIR = ""
EEG_SOURCE_FORMAT = ''
FMRI_SOURCE_FORMAT = ''
EEG_INDEX_SOURCE_FORMAT = ''
EEG_FMRI_EVENT = ''

FMRI_PROC_SCANS = []
FMRI_PROC_SCANS_TRAIN = []
FMRI_PROC_SCANS_TEST = []
FMRI_PROC_SCANS_FEWSHOT_TEST = []
FMRI_PROC_SCANS_FEWSHOT_TRAIN = []
SCAN_TASK_DICT = {}

def extract_train_windows(
    fmri_seq,
    eeg_seq,
    physio_seq,
    eeg_index_linear_raw_seq,
    eeg_index_linear_smoothed_seq,
    eeg_index_binary_total_seq,
    alpha_theta_ratio_seq,
    step_size,
    step_seg_length,
    vigilance_window_size,
    vigilance_threshold,
):
    fmri_seg, eeg_seg, physio_seg, eeg_index_linear_raw_seg, eeg_index_linear_smoothed_seg, eeg_index_binary_total_seg, alpha_theta_ratio_seg, vigilance_seg = [], [], [], [], [], [], [], []
    eeg_ratio = eeg_seq.shape[0] / fmri_seq.shape[0] 
    for start in range(0, fmri_seq.shape[0] - step_seg_length + 1, step_size):
        end = start + step_seg_length
        fmri_seg.append(fmri_seq[start:end])                   
        eeg_seg.append(eeg_seq[int(start * eeg_ratio):int(end * eeg_ratio)])              
        physio_seg.append(physio_seq[start:end])
        eeg_index_linear_raw_seg.append(eeg_index_linear_raw_seq[start:end])
        eeg_index_linear_smoothed_seg.append(eeg_index_linear_smoothed_seq[start:end])
        eeg_index_binary_total_seg.append(eeg_index_binary_total_seq[start:end])
        alpha_theta_ratio_seg.append(alpha_theta_ratio_seq[start:end])
        binary_vig_score = eeg_index_binary_total_seq[start:end].reshape(-1, vigilance_window_size)
        cluster_vig_score = binary_vig_score.sum(axis=1)
        cluster_vig_score = (cluster_vig_score > vigilance_threshold).int()
        vigilance_seg.append(cluster_vig_score)
    return (
        torch.stack(fmri_seg),          
        torch.stack(eeg_seg),
        torch.stack(physio_seg),
        torch.stack(eeg_index_linear_raw_seg),       
        torch.stack(eeg_index_linear_smoothed_seg),
        torch.stack(eeg_index_binary_total_seg),
        torch.stack(alpha_theta_ratio_seg),
        torch.stack(vigilance_seg),
    )

def smooth_moving_average_with_edge_padding(signal, window_size=5):
    signal = np.squeeze(signal)
    pad = window_size // 2
    padded_signal = np.pad(signal, pad_width=pad, mode='edge')
    smoothed = np.convolve(padded_signal, np.ones(window_size) / window_size, mode='valid')
    return smoothed


class NIHECRAlphaThetaSmallInterval1024DatasetConfig(object):
    def __init__(self):
        self.name = "NIH"

class NIHECRAlphaThetaSmallInterval1024Dataset(Dataset):
    def __init__(
            self,
            dataset_config,
            split_set="train",
            eeg_base_dir=None,
            fmri_base_dir=None,
            eeg_index_base_dir=None,
            eeg_source_format=None,
            fmri_source_format=None,
            eeg_fmri_event=None,
    ):
    
        assert split_set in ["train", "test", "val", "zero_shot", "trainval", "fewshot_train", "fewshot_test"]
        self_dataset_config = dataset_config
        
        if eeg_base_dir is None:
            eeg_base_dir=EEG_BASE_DIR
        if fmri_base_dir is None:
            fmri_base_dir=FMRI_BASE_DIR
        if eeg_index_base_dir is None:
            eeg_index_base_dir=EEG_INDEX_BASE_DIR
        if eeg_source_format is None:
            eeg_source_format = EEG_SOURCE_FORMAT
        if fmri_source_format is None:
            fmri_source_format = FMRI_SOURCE_FORMAT
        if eeg_fmri_event is None:
            eeg_fmri_event = EEG_FMRI_EVENT

        self_eeg_base_dir = eeg_base_dir
        self_eeg_mat_dir = EEG_MAT_DIR
        self_fmri_base_dir = fmri_base_dir
        self_eeg_index_base_dir = eeg_index_base_dir
        self_eeg_fmri_event = eeg_fmri_event
        self_scan_task_dict = SCAN_TASK_DICT

        all_scan_names = FMRI_PROC_SCANS
        train_scan_names = FMRI_PROC_SCANS_TRAIN
        test_scan_names = FMRI_PROC_SCANS_TEST
        fewshot_train_scan_names = FMRI_PROC_SCANS_FEWSHOT_TRAIN
        fewshot_test_scan_names = FMRI_PROC_SCANS_FEWSHOT_TEST
        
        all_subject_names = {}
        for scan_name in all_scan_names:
            subject_name = scan_name[:8]
            if subject_name not in all_subject_names:
                all_subject_names[subject_name] = []
            all_subject_names[subject_name].append(scan_name)

        train_subject_names = {}
        for scan_name in train_scan_names:
            subject_name = scan_name[:8]
            if subject_name not in train_subject_names:
                train_subject_names[subject_name] = []
            train_subject_names[subject_name].append(scan_name)
        
        test_subject_names = {}
        for scan_name in test_scan_names:
            subject_name = scan_name[:8]
            if subject_name not in test_subject_names:
                test_subject_names[subject_name] = []
            test_subject_names[subject_name].append(scan_name)
        
        if split_set == "train":
            self_scan_names = train_scan_names
            # random.shuffle(self_scan_names)
        elif split_set == "val":
            self_scan_names = train_scan_names
            # random.shuffle(self_scan_names)
        elif split_set == "trainval":
            self_scan_names = train_scan_names
            # random.shuffle(self_scan_names)
        elif split_set == "test":
            self_scan_names = test_scan_names
            # random.shuffle(self_scan_names)
        elif split_set == "zero_shot":
            self_scan_names = all_scan_names
            # random.shuffle(self_scan_names)
        elif split_set == "fewshot_train":
            self_scan_names = fewshot_train_scan_names
        elif split_set == "fewshot_test":
            self_scan_names = fewshot_test_scan_names
        
        result_scan_names = {}
        for scan_name in self_scan_names:
            subject_name = scan_name[:6]
            if subject_name in result_scan_names:
                result_scan_names[subject_name] += 1
            else:
                result_scan_names[subject_name] = 1
        
        sorted_all_subject_names = dict(sorted(all_subject_names.items()))
        sorted_result_scan_names = dict(sorted(result_scan_names.items()))
        

        print("Initializing the Dataset:")
        print("Overall subjects:")
        print(sorted_all_subject_names.keys())
        print("Successfully created the "+split_set+" dataset, with item names:")
        print(sorted(self_scan_names))
        print("With " +split_set+" subjects:")
        print(sorted_result_scan_names)  

        self.scan_names = self_scan_names 
        fmri_data_total = []
        eeg_data_total = []
        physio_data_total = []
        eeg_index_full_total = []
        eeg_index_linear_raw_total = []
        eeg_index_linear_smoothed_total = []
        eeg_index_binary_total = []
        alpha_theta_ratio_total = []
        
        def calculate_alpha_band_power(data, sf, window_sec=2.1, relative=False):
            band = [8, 12]
            freqs, psd = signal.welch(data, sf, nperseg=sf * window_sec, noverlap=0)
            band_idx = np.logical_and(freqs >= band[0], freqs <= band[1])
            band_power = np.trapz(psd[band_idx], freqs[band_idx])
            rms_amplitude = np.sqrt(band_power)
            return rms_amplitude

        def calculate_theta_band_power(data, sf, window_sec=2.1, relative=False):
            band = [3, 7]
            freqs, psd = signal.welch(data, sf, nperseg=sf * window_sec, noverlap=0)
            band_idx = np.logical_and(freqs >= band[0], freqs <= band[1])
            band_power = np.trapz(psd[band_idx], freqs[band_idx])
            rms_amplitude = np.sqrt(band_power)
            return rms_amplitude

        for idx in range(len(self_scan_names)):
            scan_name = self_scan_names[idx]
            eeg_scan_paths = glob.glob(os.path.join(self_eeg_base_dir, scan_name+'*.set'))
            eeg_scan_name = scan_name
            if len(eeg_scan_paths) == 0:
                scan_task = scan_name[17:20]
                if scan_task in self_scan_task_dict:
                    eeg_scan_name = scan_name[:17] + self_scan_task_dict[scan_task] + scan_name[20:]
                    eeg_scan_paths = glob.glob(os.path.join(self_eeg_base_dir, eeg_scan_name+'*.set'))
                if len(eeg_scan_paths) == 0:
                    eeg_scan_name = scan_name[:20] + '_' + scan_name[21:]
                    eeg_scan_paths = glob.glob(os.path.join(self_eeg_base_dir, eeg_scan_name+'*.set'))
            eeg_scan_path = eeg_scan_paths[0]
            fmri_scan_path = os.path.join(self_fmri_base_dir, scan_name+'_test_detrend_difumo_roi_1024.csv')
            scan_vigall_index = scan_name[0:12] + "{:04d}".format(int(eeg_scan_name[12:16]) + 1)
            eeg_index_path = glob.glob(os.path.join(self_eeg_index_base_dir, scan_vigall_index+'*.mat'))
            drop_list = ['time', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6']
            zero_pad_list = ['FPz', 'POz', 'FT9', 'FT10']
            change_name_dict = {"TP9": "TP9'",
                                "TP10": "TP10'"}
            eeg_data = mne.io.read_raw_eeglab(eeg_scan_path)
            df_eeg_data = eeg_data.to_data_frame()
            df_eeg_data = df_eeg_data.rename(columns=change_name_dict)
            eeg_events, eeg_events_id = mne.events_from_annotations(eeg_data)
            eeg_fmri_code = eeg_events_id[self_eeg_fmri_event]
            eeg_fmri_event = eeg_events[eeg_events[:, 2] == eeg_fmri_code]
            fmri_raw_data = pd.read_csv(fmri_scan_path)
            fmri_data_00 = fmri_raw_data.drop(columns={"Unnamed: 0"})
            # for every fMRI scan, there are 525 eeg data 
            # drop and pad columns:
            for drop_column in drop_list:
                if drop_column in df_eeg_data.columns:
                    df_eeg_data = df_eeg_data.drop([drop_column], axis=1)
            for zero_pad_column in zero_pad_list:
                if zero_pad_column not in df_eeg_data.columns:
                    df_eeg_data[zero_pad_column] = 0.0
            
            slide_cnt = (fmri_data_00.shape[0] - 2) // 5
            fmri_data = fmri_data_00.iloc[2:slide_cnt*5+2, :1024]
            first_seven_threshold = eeg_fmri_event[(7)*30][0]
            df_eeg_data = df_eeg_data.iloc[first_seven_threshold:eeg_fmri_event[30*(slide_cnt*5)][0]+first_seven_threshold, :]
            eeg_index_data_raw_total = scipy.io.loadmat(eeg_index_path[0])
            eeg_index_data_full = eeg_index_data_raw_total['VIG_SIG'][0][0][0][14:]
            eeg_index_data_linear_raw = eeg_index_data_raw_total['VIG_SIG'][0][0][1][2:slide_cnt*5+2]
            eeg_index_data_linear_smoothed = smooth_moving_average_with_edge_padding(eeg_index_data_linear_raw, window_size=5)
            eeg_index_data_binary = eeg_index_data_raw_total['VIG_SIG'][0][0][3][2:slide_cnt*5+2]
            
            alertness_channels = ['P3', 'P4', 'Pz', 'O1', 'O2', 'Oz']
            selected_channels = df_eeg_data[alertness_channels]
            averaged_signal = selected_channels.mean(axis=1)

            fs = 250
            TR = 2.1  
            samples_per_TR = int(TR * fs)
            num_TRs = averaged_signal.shape[0] // samples_per_TR
            num_samples = averaged_signal.shape[0]
            alpha_power = []
            theta_power = []
            for start in range(0, num_samples - samples_per_TR + 1, samples_per_TR):
                window = averaged_signal[start:start + samples_per_TR] # (525,)
                alpha_individual_power = calculate_alpha_band_power(window, fs)
                alpha_power.append(alpha_individual_power)
                theta_individual_power = calculate_theta_band_power(window, fs)
                theta_power.append(theta_individual_power)
            alpha_power = np.array(alpha_power)
            theta_power = np.array(theta_power)
            epsilon = 1e-10  # small value to avoid division by zero
            alpha_theta_ratio = alpha_power / (theta_power + epsilon)

            temp_fmri_data_raw = np.array(fmri_data)
            scaler = StandardScaler()
            temp_fmri_data = scaler.fit_transform(temp_fmri_data_raw) 
            temp_eeg_data = np.array(df_eeg_data)
            temp_physio = np.array(temp_fmri_data_raw)
            eeg_index_data_full = np.array(eeg_index_data_full)
            eeg_index_data_linear_raw = np.array(eeg_index_data_linear_raw)
            eeg_index_data_linear_smoothed = np.array(eeg_index_data_linear_smoothed)
            eeg_index_data_binary = np.array(eeg_index_data_binary)
            
            
            def zero_pad_to_target_length(arr, target_length):
                current_length = arr.shape[0]
                if current_length >= target_length:
                    return arr[:target_length]  # trim if longer
                pad_len = target_length - current_length
                return np.pad(arr, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
            
            target_length = 1456
            eeg_index_data_full = zero_pad_to_target_length(eeg_index_data_full, target_length)

            fmri_data_total.append(temp_fmri_data)
            eeg_data_total.append(temp_eeg_data)
            physio_data_total.append(temp_physio)
            eeg_index_full_total.append(eeg_index_data_full)
            eeg_index_linear_raw_total.append(eeg_index_data_linear_raw)
            eeg_index_linear_smoothed_total.append(eeg_index_data_linear_smoothed)
            eeg_index_binary_total.append(eeg_index_data_binary)
            smoothed_alpha_theta = smooth_moving_average_with_edge_padding(alpha_theta_ratio, window_size=5)
            alpha_theta_ratio_total.append(smoothed_alpha_theta)
        
        eeg_data = np.stack(eeg_data_total, axis=0)
        fmri_data = np.stack(fmri_data_total, axis=0)
        physio_data = np.stack(physio_data_total, axis=0)
        eeg_index_full_total = np.stack(eeg_index_full_total, axis=0)
        eeg_index_linear_raw_total = np.stack(eeg_index_linear_raw_total, axis=0)
        eeg_index_linear_smoothed_total = np.stack(eeg_index_linear_smoothed_total, axis=0)
        eeg_index_binary_total = np.stack(eeg_index_binary_total, axis=0)
        alpha_theta_ratio_total = np.stack(alpha_theta_ratio_total, axis=0)
        
        self.eeg_data = eeg_data
        self.fmri_data = fmri_data
        self.physio_data = physio_data
        self.eeg_index_full_total = eeg_index_full_total
        self.eeg_index_linear_raw_total = eeg_index_linear_raw_total
        self.eeg_index_linear_smoothed_total = eeg_index_linear_smoothed_total
        self.eeg_index_binary_total = eeg_index_binary_total
        self.alpha_theta_ratio_total = alpha_theta_ratio_total
        print(f"self.eeg_data: {self.eeg_data.shape}")
        print(f"self.fmri_data: {self.fmri_data.shape}")

        from scipy import interpolate
        def linear_interpolation_impute(data):
            data_imputed = np.copy(data)
            for scan in range(data.shape[0]):
                for signal in range(data.shape[2]):
                    series = data[scan, :, signal]
                    if np.any(np.isnan(series)):
                        time = np.arange(len(series))
                        mask = ~np.isnan(series)
                        interp_func = interpolate.interp1d(time[mask], series[mask], bounds_error=False, fill_value="extrapolate")
                        data_imputed[scan, :, signal] = interp_func(time)
            return data_imputed

        physio_imputed = linear_interpolation_impute(physio_data)
        print(f"self.physio_data: {self.physio_data.shape}")
        self.physio_data = physio_imputed

        print(f"self.eeg_index_full_total: {self.eeg_index_full_total.shape}")
        print(f"self.eeg_index_linear_raw_total: {self.eeg_index_linear_raw_total.shape}")
        print(f"self.eeg_index_linear_smoothed_total: {self.eeg_index_linear_smoothed_total.shape}")
        print(f"self.eeg_index_binary_total: {self.eeg_index_binary_total.shape}")
        print(f"self.alpha_theta_ratio_total: {self.alpha_theta_ratio_total.shape}")

        self.windowed_fmri = []
        self.windowed_eeg = []
        self.windowed_physio = []
        self.windowed_eeg_index_linear_raw_total = []
        self.windowed_eeg_index_linear_smoothed_total = []
        self.windowed_eeg_index_binary_total = []
        self.windowed_alpha_theta_ratio_total = []
        self.windowed_vigilance_seg = []

        step_size = 5
        if split_set == "train":
            step_size = 5
        step_seg_length = 5
        vigilance_window_size = 5
        vigilance_threshold = -1

        for idx in range(len(self.fmri_data)):
            fmri_seq = torch.tensor(self.fmri_data[idx], dtype=torch.float32)
            eeg_seq = torch.tensor(self.eeg_data[idx], dtype=torch.float32)
            physio_seq = torch.tensor(self.physio_data[idx], dtype=torch.float32)
            eeg_index_linear_raw_seq = torch.tensor(self.eeg_index_linear_raw_total[idx], dtype=torch.float32)
            eeg_index_linear_smoothed_seq = torch.tensor(self.eeg_index_linear_smoothed_total[idx], dtype=torch.float32)
            eeg_index_binary_total_seq = torch.tensor(self.eeg_index_binary_total[idx], dtype=torch.float32)
            alpha_theta_ratio_seq = torch.tensor(self.alpha_theta_ratio_total[idx], dtype=torch.float32)
            fmri_seg, eeg_seg, physio_seg, eeg_index_linear_raw_seg, eeg_index_linear_smoothed_seg, eeg_index_binary_total_seg, alpha_theta_ratio_seg, vigilance_seg = extract_train_windows(
                fmri_seq=fmri_seq,
                eeg_seq=eeg_seq,
                physio_seq=physio_seq,
                eeg_index_linear_raw_seq=eeg_index_linear_raw_seq,
                eeg_index_linear_smoothed_seq=eeg_index_linear_smoothed_seq,
                eeg_index_binary_total_seq=eeg_index_binary_total_seq,
                alpha_theta_ratio_seq=alpha_theta_ratio_seq,
                step_size=step_size,
                step_seg_length=step_seg_length,
                vigilance_window_size=vigilance_window_size,
                vigilance_threshold=vigilance_threshold,
            )
            self.windowed_fmri.append(fmri_seg)
            self.windowed_eeg.append(eeg_seg)
            self.windowed_physio.append(physio_seg)
            self.windowed_eeg_index_linear_raw_total.append(eeg_index_linear_raw_seg)
            self.windowed_eeg_index_linear_smoothed_total.append(eeg_index_linear_smoothed_seg)
            self.windowed_eeg_index_binary_total.append(eeg_index_binary_total_seg)
            self.windowed_alpha_theta_ratio_total.append(alpha_theta_ratio_seg)
            self.windowed_vigilance_seg.append(vigilance_seg)

        self.windowed_fmri = np.concatenate(self.windowed_fmri, axis=0)
        self.windowed_eeg = np.concatenate(self.windowed_eeg, axis=0)
        self.windowed_physio = np.concatenate(self.windowed_physio, axis=0)
        self.windowed_eeg_index_linear_raw_total = np.concatenate(self.windowed_eeg_index_linear_raw_total, axis=0)
        self.windowed_eeg_index_linear_smoothed_total = np.concatenate(self.windowed_eeg_index_linear_smoothed_total, axis=0)
        self.windowed_eeg_index_binary_total = np.concatenate(self.windowed_eeg_index_binary_total, axis=0)
        self.windowed_alpha_theta_ratio_total = np.concatenate(self.windowed_alpha_theta_ratio_total, axis=0)
        self.windowed_vigilance_seg = np.concatenate(self.windowed_vigilance_seg, axis=0)
        print(f"self.windowed_fmri: {self.windowed_fmri.shape}") 
        print(f"self.windowed_eeg: {self.windowed_eeg.shape}")
        print(f"self.windowed_physio: {self.windowed_physio.shape}")
        print(f"self.windowed_eeg_index_linear_raw_total: {self.windowed_eeg_index_linear_raw_total.shape}")
        print(f"self.windowed_eeg_index_linear_smoothed_total: {self.windowed_eeg_index_linear_smoothed_total.shape}")
        print(f"self.windowed_eeg_index_binary_total: {self.windowed_eeg_index_binary_total.shape}")
        print(f"self.windowed_alpha_theta_ratio_total: {self.windowed_alpha_theta_ratio_total.shape}")
        print(f"self.windowed_vigilance_seg: {self.windowed_vigilance_seg.shape}")


    def __len__(self):
        return len(self.windowed_fmri)

    def __getitem__(self, idx):
        return self.windowed_fmri[idx], self.windowed_eeg[idx], self.windowed_physio[idx], self.windowed_eeg_index_linear_raw_total[idx], self.windowed_eeg_index_linear_smoothed_total[idx], self.windowed_eeg_index_binary_total[idx], self.windowed_alpha_theta_ratio_total[idx], self.windowed_vigilance_seg[idx]