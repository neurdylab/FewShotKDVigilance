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

FMRI_BASE_DIR = ""
FMRI_SAVE_DIR = ''
FMRI_COORD_PATH = ""

FMRI_PROC_DIRS = []

FMRI_SOURCE_FORMAT = ''

def extract_train_windows(
    fmri_seq,
    step_size,
    step_seg_length,
):
    fmri_seg = []
    for start in range(0, fmri_seq.shape[0] - step_seg_length + 1, step_size):
        end = start + step_seg_length
        fmri_seg.append(fmri_seq[start:end])                   
    return torch.stack(fmri_seg)      


class EEGfMRIVuPatAlphaThetaSmallInterval1024DatasetConfig(object):
    def __init__(self):
        self.name = "eegfmri_vu"

class EEGfMRIVuPatAlphaThetaSmallInterval1024Dataset(Dataset):
    def __init__(
            self,
            dataset_config,
            split_set="train",
    ):
    
        assert split_set in ["train", "test", "val", "zero_shot", "trainval", "fewshot_train", "fewshot_test"]
        
        fmri_base_dir=FMRI_BASE_DIR
        fmri_proc_dirs = FMRI_PROC_DIRS
        fmri_save_dir = FMRI_SAVE_DIR
        fmri_source_format = FMRI_SOURCE_FORMAT

        self_fmri_base_dir = fmri_base_dir
        self_fmri_save_dir = fmri_save_dir

        all_scan_names = []
        all_subject_names = {}
        for fmri_dir in fmri_proc_dirs:
            fmri_data_paths = glob.glob(os.path.join(fmri_base_dir, fmri_dir, fmri_save_dir, fmri_source_format))
            for fmri_data_path in fmri_data_paths:
                scan_name = os.path.basename(fmri_data_path)[:13]
                all_scan_names.append(scan_name)
                subject_name = scan_name[:6]
                if subject_name not in all_subject_names:
                    all_subject_names[subject_name] = []
                all_subject_names[subject_name].append(scan_name)
        all_scan_names.sort()

        if split_set == "zero_shot":
            self_scan_names = all_scan_names
        
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

        for idx in range(len(self_scan_names)):
            scan_name = self_scan_names[idx]
            subject_name = scan_name[:6]
            subject_scan_name = scan_name[7:]
            fmri_scan_path = os.path.join(self_fmri_base_dir, subject_name, 'meica_proc_'+subject_scan_name, self_fmri_save_dir, scan_name+'_1024_difumo_roi.csv')
            # move 2 fMRI scans forward to account for hemodynamic response
            fmri_raw_data = pd.read_csv(fmri_scan_path)
            fmri_data_00 = fmri_raw_data.drop(columns={"Unnamed: 0"})
            # print(f"df_eeg_data.columns: {df_eeg_data.columns}")
            slide_cnt = (fmri_data_00.shape[0] - 2) // 5
            fmri_data = fmri_data_00.iloc[2:slide_cnt*5+2, :1024]
            temp_fmri_data_raw = np.array(fmri_data)
            scaler = StandardScaler()
            temp_fmri_data = scaler.fit_transform(temp_fmri_data_raw) 
            fmri_data_total.append(temp_fmri_data)
        
        fmri_data = np.stack(fmri_data_total, axis=0)
        self.fmri_data = fmri_data
        print(f"self.fmri_data: {self.fmri_data.shape}")

        self.windowed_fmri = []

        step_size = 5
        if split_set == "train":
            step_size = 5
        step_seg_length = 5

        for idx in range(len(self.fmri_data)):
            fmri_seq = torch.tensor(self.fmri_data[idx], dtype=torch.float32)
            fmri_seg = extract_train_windows(
                fmri_seq=fmri_seq,
                step_size=step_size,
                step_seg_length=step_seg_length,
            )
            self.windowed_fmri.append(fmri_seg)
        self.windowed_fmri = np.concatenate(self.windowed_fmri, axis=0)

    def __len__(self):
        return len(self.windowed_fmri)

    def __getitem__(self, idx):
        return self.windowed_fmri[idx]