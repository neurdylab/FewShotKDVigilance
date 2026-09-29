import numpy as np
import os
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from scipy import signal
import pandas as pd

NIH_ECR_SCANS = []
nih_ecr_scan_list = {
    "zero_shot": NIH_ECR_SCANS
}
NIH_ECT_SCANS = []
nih_ect_scan_list = {
    "zero_shot": NIH_ECT_SCANS
}
VU_HEALTHY_SCANS = []
VU_HEALTHY_SCANS_TEST = []
VU_HEALTHY_SCANS_TRAIN = []
vu_healthy_scan_list = {
    "train": VU_HEALTHY_SCANS_TRAIN,
    "test": VU_HEALTHY_SCANS_TEST, 
    "zero_shot": VU_HEALTHY_SCANS
}
VU_PAT_SCANS = []
# 3 alpha ranges: 8-13, 6-10, 11-14
VU_PAT_ALPHA_RANGES = {
}
vu_pat_scan_list = {
    "zero_shot": VU_PAT_SCANS
}
scan_lists = {
    "vu_healthy": vu_healthy_scan_list,
    "vu_pat": vu_pat_scan_list,
    "nih_ecr": nih_ecr_scan_list,
    "nih_ect": nih_ect_scan_list
}


def load_npz_folder_as_dict_dataset(dataset_name):
    name_to_root_dir = {
        "vu_pat": ,#TODO
        "vu_healthy": ,#TODO
        "nih_ecr": ,#TODO"
        "nih_ect": #TODO
    }
    
    name_to_eegband_root_dir = {
        "vu_pat": , #TODO
        "vu_healthy": , #TODO
        "nih_ecr": , #TODO
        "nih_ect": , #TODO
    }
    root_dir = name_to_root_dir[dataset_name]
    eeg_root_dir = name_to_eegband_root_dir[dataset_name]
    data = {}
    tr = 2.1
    files = sorted([f for f in os.listdir(root_dir) if f.endswith(".npz")])
    files_eeg = sorted([f for f in os.listdir(eeg_root_dir) if f.endswith(".npz")])
    for f in files:
        scan_name = f[:-4]
        if dataset_name == "vu_pat":
            if scan_name not in VU_PAT_SCANS:
                continue
        path = os.path.join(root_dir, f)
        with np.load(path, allow_pickle=True) as d:
            T = d["fmri_data"].shape[0]
            fmri_time = (np.arange(T, dtype=np.float32) / T)
            T_eeg = d["eeg_data"].shape[0]
            eeg_time = (np.arange(T_eeg, dtype=np.float32) / T_eeg)
            
            ratio = T_eeg // T
            eeg_index_binary_eeg_rate = np.repeat(d["eeg_index_binary"], ratio)
            data[scan_name] = {
                "scan_name": d["scan_name"].item() if d["scan_name"].shape == () else d["scan_name"],
                "eeg_data": d["eeg_data"],
                "eeg_data_columns": d["eeg_data_columns"],
                "fmri_data": d["fmri_data"],
                "fmri_data_columns": d["fmri_data_columns"],
                "eeg_index_linear_raw": d["eeg_index_linear_raw"],          
                "eeg_index_linear_smoothed": d["eeg_index_linear_smoothed"],
                "eeg_index_binary": d["eeg_index_binary"],  
                "eeg_index_binary_smoothed": gaussian_filter1d(d["eeg_index_binary"].astype(np.float32), sigma=5), 
                "eeg_index_binary_eeg_rate": eeg_index_binary_eeg_rate,
                "fmri_time": fmri_time,
                "eeg_time": eeg_time,
            }
    
    for f in files_eeg:
        scan_name = f[:-4]
        if scan_name not in data.keys():
            continue
        path = os.path.join(eeg_root_dir, f)
        with np.load(path, allow_pickle=True) as d:
            data[scan_name]["alpha_power"] = d["alpha_power"]
            data[scan_name]["alpha_smoothed"] = zscore(gaussian_filter1d(d["alpha_power"], sigma=3))
            data[scan_name]["theta_power"] = d["theta_power"]
            data[scan_name]["theta_smoothed"] = zscore(gaussian_filter1d(d["theta_power"], sigma=3))
            data[scan_name]["delta_power"] = d["delta_power"]
            data[scan_name]["delta_smoothed"] = zscore(gaussian_filter1d(d["delta_power"], sigma=3)) 
            data[scan_name]["delta_theta_power"] = d["delta_theta_power"]
            data[scan_name]["delta_theta_smoothed"] = zscore(gaussian_filter1d(d["delta_theta_power"], sigma=3)) 
            data[scan_name]["beta_power"] = d["beta_power"]
            data[scan_name]["beta_smoothed"] = zscore(gaussian_filter1d(d["beta_power"], sigma=3)) 
            data[scan_name]["tot_power"] = d["tot_power"]
            data[scan_name]["tot_smoothed"] = zscore(gaussian_filter1d(d["tot_power"], sigma=3)) 
            data[scan_name]["alpha_theta_ratio"] = d["alpha_theta_ratio"]
            data[scan_name]["alpha_theta_ratio_smoothed"] = zscore(gaussian_filter1d(d["alpha_theta_ratio"], sigma=3))
            data[scan_name]["beta_theta_ratio"] = d["beta_theta_ratio"]
            data[scan_name]["beta_theta_ratio_smoothed"] = zscore(gaussian_filter1d(d["beta_theta_ratio"], sigma=3)) 
            data[scan_name]["alpha_deltatheta_ratio"] = d["alpha_deltatheta_ratio"]
            data[scan_name]["alpha_deltatheta_ratio_smoothed"] = zscore(gaussian_filter1d(d["alpha_deltatheta_ratio"], sigma=3)) 
            data[scan_name]["beta_deltatheta_ratio"] = d["beta_deltatheta_ratio"]
            data[scan_name]["beta_deltatheta_ratio_smoothed"] = zscore(gaussian_filter1d(d["beta_deltatheta_ratio"], sigma=3))  
            

            if dataset_name == "vu_pat":
                if scan_name not in VU_PAT_SCANS:
                    continue
                alpha_range = VU_PAT_ALPHA_RANGES[scan_name]
        
                def calculate_band_power(data, band, sf, window_sec=2.1, relative=False):
                    freqs, psd = signal.welch(data, sf, nperseg=sf * window_sec, noverlap=0)
                    band_idx = np.logical_and(freqs >= band[0], freqs <= band[1])
                    band_power = np.trapz(psd[band_idx], freqs[band_idx])
                    return band_power
                                    
                alertness_channels = ['P3', 'P4', 'Pz', 'O1', 'O2', 'Oz']
                scan_data = data[scan_name]
                scan_eeg = scan_data['eeg_data']
                scan_eeg_columns = scan_data['eeg_data_columns']
                df_eeg_data = pd.DataFrame(scan_eeg, columns=scan_eeg_columns)
                selected_channels = df_eeg_data[alertness_channels]
                averaged_signal = selected_channels.mean(axis=1)
                fs = 250
                TR = 2.1  
                samples_per_TR = int(TR * fs)
                num_TRs = averaged_signal.shape[0] // samples_per_TR
                num_samples = averaged_signal.shape[0]
                alpha_power = []
                for start in range(0, num_samples - samples_per_TR + 1, samples_per_TR):
                    window = averaged_signal[start:start + samples_per_TR] 
                    alpha_individual_power = calculate_band_power(window, [alpha_range[0], alpha_range[1]], fs) 
                    alpha_power.append(alpha_individual_power)
                alpha_power = np.array(alpha_power)

                x_orig = np.linspace(0, 1, 570)
                x_target = np.linspace(0, 1, 575)
                data[scan_name]["alpha_power"] = np.sqrt(np.interp(x_target, x_orig, alpha_power))

                eps = 1e-8
                data[scan_name]["alpha_tot_ratio"] = data[scan_name]["alpha_power"]/(data[scan_name]["tot_power"] + eps)
                data[scan_name]["theta_tot_ratio"] = data[scan_name]["theta_power"]/(data[scan_name]["tot_power"] + eps)
                data[scan_name]["delta_tot_ratio"] = data[scan_name]["delta_power"]/(data[scan_name]["tot_power"] + eps)
                data[scan_name]["beta_tot_ratio"] = data[scan_name]["beta_power"]/(data[scan_name]["tot_power"] + eps)
                
                data[scan_name]["alpha_tot_ratio_smoothed"] = gaussian_filter1d(data[scan_name]["alpha_tot_ratio"], sigma=3)
                data[scan_name]["theta_tot_ratio_smoothed"] = gaussian_filter1d(data[scan_name]["theta_tot_ratio"], sigma=3)
                data[scan_name]["delta_tot_ratio_smoothed"] = gaussian_filter1d(data[scan_name]["delta_tot_ratio"], sigma=3)
                data[scan_name]["beta_tot_ratio_smoothed"] = gaussian_filter1d(data[scan_name]["beta_tot_ratio"], sigma=3)
                
            else:
                eps = 1e-8
                data[scan_name]["alpha_tot_ratio"] = data[scan_name]["alpha_power"]/(data[scan_name]["tot_power"] + eps)
                data[scan_name]["theta_tot_ratio"] = data[scan_name]["theta_power"]/(data[scan_name]["tot_power"] + eps)
                data[scan_name]["delta_tot_ratio"] = data[scan_name]["delta_power"]/(data[scan_name]["tot_power"] + eps)
                data[scan_name]["beta_tot_ratio"] = data[scan_name]["beta_power"]/(data[scan_name]["tot_power"] + eps)
                
                data[scan_name]["alpha_tot_ratio_smoothed"] = gaussian_filter1d(data[scan_name]["alpha_tot_ratio"], sigma=3)
                data[scan_name]["theta_tot_ratio_smoothed"] = gaussian_filter1d(data[scan_name]["theta_tot_ratio"], sigma=3)
                data[scan_name]["delta_tot_ratio_smoothed"] = gaussian_filter1d(data[scan_name]["delta_tot_ratio"], sigma=3)
                data[scan_name]["beta_tot_ratio_smoothed"] = gaussian_filter1d(data[scan_name]["beta_tot_ratio"], sigma=3)
    return data


def eegfmri_allocation(dataset_name):
    dataset = load_npz_folder_as_dict_dataset(dataset_name)
    dataset_partition = scan_lists[dataset_name]
    return dataset, dataset_partition


def build_set(dataset, 
              scan_names, 
              window_size=5, 
              step_size=5,
              vigilance_threshold=-1, 
              eeg_rate=525):
    
    eeg_data_list = []
    eeg_data_columns_list = []
    fmri_data_list = []
    fmri_data_columns_list = []
    eeg_index_linear_raw_list = []
    eeg_index_linear_smoothed_list = []
    eeg_index_binary_list = []
    eeg_index_binary_smoothed_list = []
    eeg_index_binary_eeg_rate_list = []
    fmri_time_list = []
    eeg_time_list = []
    alpha_power_list = []
    alpha_smoothed_list = []
    theta_power_list = []
    theta_smoothed_list = []
    delta_power_list = []
    delta_smoothed_list = []
    delta_theta_power_list = []
    delta_theta_smoothed_list = []
    beta_power_list = []
    beta_smoothed_list = []
    tot_power_list = []
    tot_smoothed_list = []
    alpha_theta_ratio_list = []
    alpha_theta_ratio_smoothed_list = []
    beta_theta_ratio_list = []
    beta_theta_ratio_smoothed_list = []
    alpha_deltatheta_ratio_list = []
    alpha_deltatheta_ratio_smoothed_list = []
    beta_deltatheta_ratio_list = []
    beta_deltatheta_ratio_smoothed_list = []
    
    alpha_tot_ratio_list = []
    theta_tot_ratio_list = []
    delta_tot_ratio_list = []
    beta_tot_ratio_list = []
    alpha_tot_ratio_smoothed_list = []
    theta_tot_ratio_smoothed_list = []
    delta_tot_ratio_smoothed_list = []
    beta_tot_ratio_smoothed_list = []
    
    for scan_name in scan_names:
        d = dataset[scan_name]
        eeg_data_list.append(d["eeg_data"])
        eeg_data_columns_list.append(d["eeg_data_columns"])
        fmri_data_list.append(d["fmri_data"])
        time_span = d["fmri_data"].shape[0]
        fmri_data_columns_list.append(d["fmri_data_columns"])
        eeg_index_linear_raw_list.append(d["eeg_index_linear_raw"])
        eeg_index_linear_smoothed_list.append(d["eeg_index_linear_smoothed"])
        eeg_index_binary_list.append(d["eeg_index_binary"])
        eeg_index_binary_smoothed_list.append(d["eeg_index_binary_smoothed"])
        eeg_index_binary_eeg_rate_list.append(d["eeg_index_binary_eeg_rate"])
        fmri_time_list.append(d["fmri_time"])
        eeg_time_list.append(d["eeg_time"])
        alpha_power_list.append(d["alpha_power"][2:time_span+2])
        alpha_smoothed_list.append(d["alpha_smoothed"][2:time_span+2])
        theta_power_list.append(d["theta_power"][2:time_span+2])
        theta_smoothed_list.append(d["theta_smoothed"][2:time_span+2])
        delta_power_list.append(d["delta_power"][2:time_span+2])
        delta_smoothed_list.append(d["delta_smoothed"][2:time_span+2])
        delta_theta_power_list.append(d["delta_theta_power"][2:time_span+2])
        delta_theta_smoothed_list.append(d["delta_theta_smoothed"][2:time_span+2])
        beta_power_list.append(d["beta_power"][2:time_span+2])
        beta_smoothed_list.append(d["beta_smoothed"][2:time_span+2])
        tot_power_list.append(d["tot_power"][2:time_span+2])
        tot_smoothed_list.append(d["tot_smoothed"][2:time_span+2])
        alpha_theta_ratio_list.append(d["alpha_theta_ratio"][2:time_span+2])
        alpha_theta_ratio_smoothed_list.append(d["alpha_theta_ratio_smoothed"][2:time_span+2])
        beta_theta_ratio_list.append(d["beta_theta_ratio"][2:time_span+2])
        beta_theta_ratio_smoothed_list.append(d["beta_theta_ratio_smoothed"][2:time_span+2])
        alpha_deltatheta_ratio_list.append(d["alpha_deltatheta_ratio"][2:time_span+2])
        alpha_deltatheta_ratio_smoothed_list.append(d["alpha_deltatheta_ratio_smoothed"][2:time_span+2])
        beta_deltatheta_ratio_list.append(d["beta_deltatheta_ratio"][2:time_span+2])
        beta_deltatheta_ratio_smoothed_list.append(d["beta_deltatheta_ratio_smoothed"][2:time_span+2])
        
        alpha_tot_ratio_list.append(d["alpha_tot_ratio"][2:time_span+2])
        theta_tot_ratio_list.append(d["theta_tot_ratio"][2:time_span+2])
        delta_tot_ratio_list.append(d["delta_tot_ratio"][2:time_span+2])
        beta_tot_ratio_list.append(d["beta_tot_ratio"][2:time_span+2])
        
        alpha_tot_ratio_smoothed_list.append(d["alpha_tot_ratio_smoothed"][2:time_span+2])
        theta_tot_ratio_smoothed_list.append(d["theta_tot_ratio_smoothed"][2:time_span+2])
        delta_tot_ratio_smoothed_list.append(d["delta_tot_ratio_smoothed"][2:time_span+2])
        beta_tot_ratio_smoothed_list.append(d["beta_tot_ratio_smoothed"][2:time_span+2])

    eeg_data = np.lib.stride_tricks.sliding_window_view(
        np.stack(eeg_data_list, axis=0), 
        window_shape=(window_size * eeg_rate,), 
        axis=1
    )[:, ::step_size * eeg_rate, :]
    eeg_data = eeg_data.reshape(-1, eeg_data.shape[2], eeg_data.shape[3]) # (798, 26, 2625)
    
    fmri_data = np.lib.stride_tricks.sliding_window_view(
        np.stack(fmri_data_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size, :]
    fmri_data = fmri_data.reshape(-1, fmri_data.shape[2], fmri_data.shape[3]) # (798, 66, 5)
    
    eeg_index_linear_raw = np.lib.stride_tricks.sliding_window_view(
        np.stack(eeg_index_linear_raw_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size] # (7, 114, 5)
    eeg_index_linear_raw = eeg_index_linear_raw.reshape(-1, eeg_index_linear_raw.shape[2])  # (798, 5)
    
    eeg_index_linear_smoothed = np.lib.stride_tricks.sliding_window_view(
        np.stack(eeg_index_linear_smoothed_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size] 
    eeg_index_linear_smoothed = eeg_index_linear_smoothed.reshape(-1, eeg_index_linear_smoothed.shape[2]) 

    eeg_index_binary = np.lib.stride_tricks.sliding_window_view(
        np.stack(eeg_index_binary_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size] 
    eeg_index_binary = eeg_index_binary.reshape(-1, eeg_index_binary.shape[2])  
    
    gt_total = (np.sum(eeg_index_binary, axis=1) > vigilance_threshold).astype(int).reshape(-1, 1)
    
    eeg_index_linear_smoothed = np.lib.stride_tricks.sliding_window_view(
        np.stack(eeg_index_linear_smoothed_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size] 
    eeg_index_linear_smoothed = eeg_index_linear_smoothed.reshape(-1, eeg_index_linear_smoothed.shape[2]) 
    
    eeg_index_binary_eeg_rate = np.lib.stride_tricks.sliding_window_view(
        np.stack(eeg_index_binary_eeg_rate_list, axis=0),
        window_shape=window_size*eeg_rate,
        axis=1
    )[:, ::step_size*eeg_rate] 
    eeg_index_binary_eeg_rate = eeg_index_binary_eeg_rate.reshape(-1, eeg_index_binary_eeg_rate.shape[2]) 
    
    fmri_time = np.lib.stride_tricks.sliding_window_view(
        np.stack(fmri_time_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    fmri_time = fmri_time.reshape(-1, fmri_time.shape[2])

    eeg_time = np.lib.stride_tricks.sliding_window_view(
        np.stack(eeg_time_list, axis=0),
        window_shape=(window_size * eeg_rate,),
        axis=1
    )[:, ::step_size * eeg_rate]
    eeg_time = eeg_time.reshape(-1, eeg_time.shape[2])

    alpha_power = np.lib.stride_tricks.sliding_window_view(
        np.stack(alpha_power_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    alpha_power = alpha_power.reshape(-1, alpha_power.shape[2])

    alpha_smoothed = np.lib.stride_tricks.sliding_window_view(
        np.stack(alpha_smoothed_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    alpha_smoothed = alpha_smoothed.reshape(-1, alpha_smoothed.shape[2])

    theta_power = np.lib.stride_tricks.sliding_window_view(
        np.stack(theta_power_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    theta_power = theta_power.reshape(-1, theta_power.shape[2])

    theta_smoothed = np.lib.stride_tricks.sliding_window_view(
        np.stack(theta_smoothed_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    theta_smoothed = theta_smoothed.reshape(-1, theta_smoothed.shape[2])
    
    delta_power = np.lib.stride_tricks.sliding_window_view(
        np.stack(delta_power_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    delta_power = delta_power.reshape(-1, delta_power.shape[2])

    delta_smoothed = np.lib.stride_tricks.sliding_window_view(
        np.stack(delta_smoothed_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    delta_smoothed = delta_smoothed.reshape(-1, delta_smoothed.shape[2])
    
    delta_theta_power = np.lib.stride_tricks.sliding_window_view(
        np.stack(delta_theta_power_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    delta_theta_power = delta_theta_power.reshape(-1, delta_theta_power.shape[2])

    delta_theta_smoothed = np.lib.stride_tricks.sliding_window_view(
        np.stack(delta_theta_smoothed_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    delta_theta_smoothed = delta_theta_smoothed.reshape(-1, delta_theta_smoothed.shape[2])

    beta_power = np.lib.stride_tricks.sliding_window_view(
        np.stack(beta_power_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    beta_power = beta_power.reshape(-1, beta_power.shape[2])

    beta_smoothed = np.lib.stride_tricks.sliding_window_view(
        np.stack(beta_smoothed_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    beta_smoothed = beta_smoothed.reshape(-1, beta_smoothed.shape[2])

    tot_power = np.lib.stride_tricks.sliding_window_view(
        np.stack(tot_power_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    tot_power = tot_power.reshape(-1, tot_power.shape[2])

    tot_smoothed = np.lib.stride_tricks.sliding_window_view(
        np.stack(tot_smoothed_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    tot_smoothed = tot_smoothed.reshape(-1, tot_smoothed.shape[2])

    alpha_theta_ratio = np.lib.stride_tricks.sliding_window_view(
        np.stack(alpha_theta_ratio_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    alpha_theta_ratio = alpha_theta_ratio.reshape(-1, alpha_theta_ratio.shape[2])

    alpha_theta_ratio_smoothed = np.lib.stride_tricks.sliding_window_view(
        np.stack(alpha_theta_ratio_smoothed_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    alpha_theta_ratio_smoothed = alpha_theta_ratio_smoothed.reshape(-1, alpha_theta_ratio_smoothed.shape[-1])
    
    beta_theta_ratio = np.lib.stride_tricks.sliding_window_view(
        np.stack(beta_theta_ratio_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    beta_theta_ratio = beta_theta_ratio.reshape(-1, beta_theta_ratio.shape[2])

    beta_theta_ratio_smoothed = np.lib.stride_tricks.sliding_window_view(
        np.stack(beta_theta_ratio_smoothed_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    beta_theta_ratio_smoothed = beta_theta_ratio_smoothed.reshape(-1, beta_theta_ratio_smoothed.shape[-1])
    
    alpha_deltatheta_ratio = np.lib.stride_tricks.sliding_window_view(
        np.stack(alpha_deltatheta_ratio_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    alpha_deltatheta_ratio = alpha_deltatheta_ratio.reshape(-1, alpha_deltatheta_ratio.shape[2])

    alpha_deltatheta_ratio_smoothed = np.lib.stride_tricks.sliding_window_view(
        np.stack(alpha_deltatheta_ratio_smoothed_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    alpha_deltatheta_ratio_smoothed = alpha_deltatheta_ratio_smoothed.reshape(-1, alpha_deltatheta_ratio_smoothed.shape[-1])
    
    beta_deltatheta_ratio = np.lib.stride_tricks.sliding_window_view(
        np.stack(beta_deltatheta_ratio_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    beta_deltatheta_ratio = beta_deltatheta_ratio.reshape(-1, beta_deltatheta_ratio.shape[2])

    beta_deltatheta_ratio_smoothed = np.lib.stride_tricks.sliding_window_view(
        np.stack(beta_deltatheta_ratio_smoothed_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    beta_deltatheta_ratio_smoothed = beta_deltatheta_ratio_smoothed.reshape(-1, beta_deltatheta_ratio_smoothed.shape[-1])
    
    alpha_tot_ratio = np.lib.stride_tricks.sliding_window_view(
        np.stack(alpha_tot_ratio_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    alpha_tot_ratio = alpha_tot_ratio.reshape(-1, alpha_tot_ratio.shape[-1])
    
    theta_tot_ratio = np.lib.stride_tricks.sliding_window_view(
        np.stack(theta_tot_ratio_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    theta_tot_ratio = theta_tot_ratio.reshape(-1, theta_tot_ratio.shape[-1])
    
    delta_tot_ratio = np.lib.stride_tricks.sliding_window_view(
        np.stack(delta_tot_ratio_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    delta_tot_ratio = delta_tot_ratio.reshape(-1, delta_tot_ratio.shape[-1])
    
    beta_tot_ratio = np.lib.stride_tricks.sliding_window_view(
        np.stack(beta_tot_ratio_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    beta_tot_ratio = beta_tot_ratio.reshape(-1, beta_tot_ratio.shape[-1])
    
    alpha_tot_ratio_smoothed = np.lib.stride_tricks.sliding_window_view(
        np.stack(alpha_tot_ratio_smoothed_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    alpha_tot_ratio_smoothed = alpha_tot_ratio_smoothed.reshape(-1, alpha_tot_ratio_smoothed.shape[-1])
    
    theta_tot_ratio_smoothed = np.lib.stride_tricks.sliding_window_view(
        np.stack(theta_tot_ratio_smoothed_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    theta_tot_ratio_smoothed = theta_tot_ratio_smoothed.reshape(-1, theta_tot_ratio_smoothed.shape[-1])
    
    delta_tot_ratio_smoothed = np.lib.stride_tricks.sliding_window_view(
        np.stack(delta_tot_ratio_smoothed_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    delta_tot_ratio_smoothed = delta_tot_ratio_smoothed.reshape(-1, delta_tot_ratio_smoothed.shape[-1])
    
    beta_tot_ratio_smoothed = np.lib.stride_tricks.sliding_window_view(
        np.stack(beta_tot_ratio_smoothed_list, axis=0),
        window_shape=window_size,
        axis=1
    )[:, ::step_size]
    beta_tot_ratio_smoothed = beta_tot_ratio_smoothed.reshape(-1, beta_tot_ratio_smoothed.shape[-1])
    
    return_dict = {}
    return_dict["eeg_data"] = eeg_data
    return_dict["fmri_data"] = fmri_data[:, :1024]
    return_dict["eeg_index_linear_raw"] = eeg_index_linear_raw
    return_dict["eeg_index_linear_smoothed"] = eeg_index_linear_smoothed
    return_dict["eeg_index_binary"] = eeg_index_binary
    return_dict["gt_total"] = gt_total
    return_dict["eeg_index_linear_smoothed"] = eeg_index_linear_smoothed
    return_dict["eeg_index_binary_eeg_rate"] = eeg_index_binary_eeg_rate
    return_dict["fmri_time"] = fmri_time
    return_dict["eeg_time"] = eeg_time
    return_dict["alpha_power"] = alpha_power
    return_dict["alpha_smoothed"] = alpha_smoothed
    return_dict["theta_power"] = theta_power
    return_dict["theta_smoothed"] = theta_smoothed
    return_dict["delta_power"] = delta_power
    return_dict["delta_smoothed"] = delta_smoothed
    return_dict["delta_theta_power"] = delta_theta_power
    return_dict["delta_theta_smoothed"] = delta_theta_smoothed
    return_dict["beta_power"] = beta_power
    return_dict["beta_smoothed"] = beta_smoothed
    return_dict["tot_power"] = tot_power
    return_dict["tot_smoothed"] = tot_smoothed
    return_dict["alpha_theta_ratio"] = alpha_theta_ratio
    return_dict["alpha_theta_ratio_smoothed"] = alpha_theta_ratio_smoothed
    return_dict["beta_theta_ratio"] = beta_theta_ratio
    return_dict["beta_theta_ratio_smoothed"] = beta_theta_ratio_smoothed
    return_dict["alpha_deltatheta_ratio"] = alpha_deltatheta_ratio
    return_dict["alpha_deltatheta_ratio_smoothed"] = alpha_deltatheta_ratio_smoothed
    return_dict["beta_deltatheta_ratio"] = beta_deltatheta_ratio
    return_dict["beta_deltatheta_ratio_smoothed"] = beta_deltatheta_ratio_smoothed
    
    return_dict["alpha_tot_ratio"] = alpha_tot_ratio
    return_dict["theta_tot_ratio"] = theta_tot_ratio
    return_dict["delta_tot_ratio"] = delta_tot_ratio
    return_dict["beta_tot_ratio"] = beta_tot_ratio 
    
    return_dict["alpha_tot_ratio_smoothed"] = alpha_tot_ratio_smoothed
    return_dict["theta_tot_ratio_smoothed"] = theta_tot_ratio_smoothed
    return_dict["delta_tot_ratio_smoothed"] = delta_tot_ratio_smoothed
    return_dict["beta_tot_ratio_smoothed"] = beta_tot_ratio_smoothed         
    
    return return_dict, eeg_data_columns_list, fmri_data_columns_list


class VigilanceDataset1024(Dataset):
    def __init__(
            self,
            dataset_name = None,
            window_size = None,
            step_size = None,
            vigilance_threshold=None,
            split_set="train",
    ):
        eegfmri_data = load_npz_folder_as_dict_dataset(dataset_name)
        dataset_split = scan_lists[dataset_name]
        assert split_set in ["train", "test", "val", "zero_shot"]
        
        print(f"Creating VigilanceDataset1024: dataset_name: {dataset_name}; window_size: {window_size}; step_size: {step_size}; vigilance_threshold: {vigilance_threshold}; split_set: {split_set}")

        dataset, eeg_data_columns_list, fmri_data_columns_list = build_set(eegfmri_data, dataset_split[split_set], window_size, step_size, vigilance_threshold)
        
        if dataset_name is not None:
            self.dataset_name = dataset_name
        if window_size is not None:
            self.window_size = window_size
        if step_size is not None:
            self.step_size = step_size
        if vigilance_threshold is not None:
            self.vigilance_threshold = vigilance_threshold
            
        self.scan_names = dataset_split[split_set]
        
        self.eeg_data_columns_list = eeg_data_columns_list
        self.fmri_data_columns_list = fmri_data_columns_list
        self.eeg_data = dataset["eeg_data"]
        self.fmri_data = dataset["fmri_data"]
        self.eeg_index_linear_raw = dataset["eeg_index_linear_raw"]
        self.eeg_index_linear_smoothed = dataset["eeg_index_linear_smoothed"]
        self.eeg_index_binary = dataset["eeg_index_binary"]
        self.gt_total = dataset["gt_total"]
        self.eeg_index_binary_eeg_rate = dataset["eeg_index_binary_eeg_rate"]
        self.fmri_time = dataset["fmri_time"]
        self.eeg_time = dataset["eeg_time"]
        self.alpha_power = dataset["alpha_power"]
        self.alpha_smoothed = dataset["alpha_smoothed"]
        self.theta_power = dataset["theta_power"]
        self.theta_smoothed = dataset["theta_smoothed"]
        self.delta_power = dataset["delta_power"]
        self.delta_smoothed = dataset["delta_smoothed"]
        self.delta_theta_power = dataset["delta_theta_power"]
        self.delta_theta_smoothed = dataset["delta_theta_smoothed"]
        self.beta_power = dataset["beta_power"]
        self.beta_smoothed = dataset["beta_smoothed"]
        self.tot_power = dataset["tot_power"]
        self.tot_smoothed = dataset["tot_smoothed"]
        self.alpha_theta_ratio = dataset["alpha_theta_ratio"]
        self.alpha_theta_ratio_smoothed = dataset["alpha_theta_ratio_smoothed"]
        self.beta_theta_ratio = dataset["beta_theta_ratio"]
        self.beta_theta_ratio_smoothed = dataset["beta_theta_ratio_smoothed"]
        self.alpha_deltatheta_ratio = dataset["alpha_deltatheta_ratio"]
        self.alpha_deltatheta_ratio_smoothed = dataset["alpha_deltatheta_ratio_smoothed"]
        self.beta_deltatheta_ratio = dataset["beta_deltatheta_ratio"]
        self.beta_deltatheta_ratio_smoothed = dataset["beta_deltatheta_ratio_smoothed"]
        
        self.alpha_tot_ratio = dataset["alpha_tot_ratio"]
        self.theta_tot_ratio = dataset["theta_tot_ratio"]
        self.delta_tot_ratio = dataset["delta_tot_ratio"]
        self.beta_tot_ratio = dataset["beta_tot_ratio"]
        self.alpha_tot_ratio_smoothed = dataset["alpha_tot_ratio_smoothed"]
        self.theta_tot_ratio_smoothed = dataset["theta_tot_ratio_smoothed"]
        self.delta_tot_ratio_smoothed = dataset["delta_tot_ratio_smoothed"]
        self.beta_tot_ratio_smoothed = dataset["beta_tot_ratio_smoothed"]

        print(f"Final dataset length: {len(self.fmri_data)}")

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        ret_dict = {}
        ret_dict["eeg_data"] = np.array(self.eeg_data[idx])
        ret_dict["fmri_data"] = np.array(self.fmri_data[idx])
        ret_dict["eeg_index_linear_raw"] = np.array(self.eeg_index_linear_raw[idx])
        ret_dict["eeg_index_linear_smoothed"] = np.array(self.eeg_index_linear_smoothed[idx])
        ret_dict["eeg_index_binary"] = np.array(self.eeg_index_binary[idx])
        ret_dict["gt_total"] = np.array(self.gt_total[idx])
        ret_dict["eeg_index_binary_eeg_rate"] = np.array(self.eeg_index_binary_eeg_rate[idx])
        ret_dict["fmri_time"] = np.array(self.fmri_time[idx])
        ret_dict["eeg_time"] = np.array(self.eeg_time[idx])
        ret_dict["alpha_power"] = np.array(self.alpha_power[idx])
        ret_dict["alpha_smoothed"] = np.array(self.alpha_smoothed[idx])
        ret_dict["theta_power"] = np.array(self.theta_power[idx])
        ret_dict["theta_smoothed"] = np.array(self.theta_smoothed[idx])
        ret_dict["delta_power"] = np.array(self.delta_power[idx])
        ret_dict["delta_smoothed"] = np.array(self.delta_smoothed[idx])
        ret_dict["delta_theta_power"] = np.array(self.delta_theta_power[idx])
        ret_dict["delta_theta_smoothed"] = np.array(self.delta_theta_smoothed[idx])
        ret_dict["beta_power"] = np.array(self.beta_power[idx])
        ret_dict["beta_smoothed"] = np.array(self.beta_smoothed[idx])
        ret_dict["tot_power"] = np.array(self.tot_power[idx])
        ret_dict["tot_smoothed"] = np.array(self.tot_smoothed[idx])
        ret_dict["alpha_theta_ratio"] = np.array(self.alpha_theta_ratio[idx])
        ret_dict["alpha_theta_ratio_smoothed"] = np.array(self.alpha_theta_ratio_smoothed[idx])
        ret_dict["beta_theta_ratio"] = np.array(self.beta_theta_ratio[idx])
        ret_dict["beta_theta_ratio_smoothed"] = np.array(self.beta_theta_ratio_smoothed[idx])
        ret_dict["alpha_deltatheta_ratio"] = np.array(self.alpha_deltatheta_ratio[idx])
        ret_dict["alpha_deltatheta_ratio_smoothed"] = np.array(self.alpha_deltatheta_ratio_smoothed[idx])
        ret_dict["beta_deltatheta_ratio"] = np.array(self.beta_deltatheta_ratio[idx])
        ret_dict["beta_deltatheta_ratio_smoothed"] = np.array(self.beta_deltatheta_ratio_smoothed[idx])
        
        ret_dict["alpha_tot_ratio"] = np.array(self.alpha_tot_ratio[idx])
        ret_dict["theta_tot_ratio"] = np.array(self.theta_tot_ratio[idx])
        ret_dict["delta_tot_ratio"] = np.array(self.delta_tot_ratio[idx])
        ret_dict["beta_tot_ratio"] = np.array(self.beta_tot_ratio[idx])
        
        ret_dict["alpha_tot_ratio_smoothed"] = np.array(self.alpha_tot_ratio_smoothed[idx])
        ret_dict["theta_tot_ratio_smoothed"] = np.array(self.theta_tot_ratio_smoothed[idx])
        ret_dict["delta_tot_ratio_smoothed"] = np.array(self.delta_tot_ratio_smoothed[idx])
        ret_dict["beta_tot_ratio_smoothed"] = np.array(self.beta_tot_ratio_smoothed[idx])
        
        return ret_dict