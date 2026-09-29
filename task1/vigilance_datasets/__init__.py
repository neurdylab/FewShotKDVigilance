from .eegfmri_vu_alphatheta_smallinterval_1024 import EEGfMRIVuAlphaThetaSmallInterval1024Dataset, EEGfMRIVuAlphaThetaSmallInterval1024DatasetConfig
from .eegfmri_vu_pat_alphatheta_smallinterval_1024 import EEGfMRIVuPatAlphaThetaSmallInterval1024Dataset, EEGfMRIVuPatAlphaThetaSmallInterval1024DatasetConfig
from .eegfmri_vu_pat_alphatheta_smallinterval_1024_gt import EEGfMRIVuPatAlphaThetaSmallInterval1024GTDataset, EEGfMRIVuPatAlphaThetaSmallInterval1024GTDatasetConfig
from .NIHECR_alphatheta_smallinterval_1024 import NIHECRAlphaThetaSmallInterval1024Dataset, NIHECRAlphaThetaSmallInterval1024DatasetConfig

DATASET_FUNCTIONS = {
    "eegfmri_vu_alphatheta_smallinterval_1024": [EEGfMRIVuAlphaThetaSmallInterval1024Dataset, EEGfMRIVuAlphaThetaSmallInterval1024DatasetConfig],
    "eegfmri_vu_pat_alphatheta_smallinterval_1024": [EEGfMRIVuPatAlphaThetaSmallInterval1024Dataset, EEGfMRIVuPatAlphaThetaSmallInterval1024DatasetConfig],
    "eegfmri_vu_pat_alphatheta_smallinterval_1024_gt": [EEGfMRIVuPatAlphaThetaSmallInterval1024GTDataset, EEGfMRIVuPatAlphaThetaSmallInterval1024GTDatasetConfig],
    "NIHECR_alphatheta_smallinterval_1024": [NIHECRAlphaThetaSmallInterval1024Dataset, NIHECRAlphaThetaSmallInterval1024DatasetConfig],
}

def build_dataset(dataset_name):
    dataset_builder = DATASET_FUNCTIONS[dataset_name][0]
    dataset_config = DATASET_FUNCTIONS[dataset_name][1]()
    train_dataset = dataset_builder(dataset_config,
            split_set="train",)
    test_dataset = dataset_builder(dataset_config,
            split_set="test",)
    return train_dataset, test_dataset

def build_dataset_zeroshot(dataset_name):
    dataset_builder = DATASET_FUNCTIONS[dataset_name][0]
    dataset_config = DATASET_FUNCTIONS[dataset_name][1]()
    zeroshot_dataset = dataset_builder(dataset_config,
            split_set="zero_shot",)
    return zeroshot_dataset

def build_dataset_fewshot(dataset_name):
    dataset_builder = DATASET_FUNCTIONS[dataset_name][0]
    dataset_config = DATASET_FUNCTIONS[dataset_name][1]()
    fewshot_train_dataset = dataset_builder(dataset_config,
            split_set="fewshot_train",)
    fewshot_test_dataset = dataset_builder(dataset_config,
            split_set="fewshot_test",)
    return fewshot_train_dataset, fewshot_test_dataset


def build_dataset_prev(args):
    dataset_builder = DATASET_FUNCTIONS[args.dataset][0]
    dataset_config = DATASET_FUNCTIONS[args.dataset][1]()
    dataset_dict = {
        "train": dataset_builder(
            dataset_config,
            split_set="train",
        ),
        "test": dataset_builder(
            dataset_config,
            split_set="test",
        ),
        "zero_shot": dataset_builder(
            dataset_config,
            split_set="zero_shot",
        ),
    }
    return dataset_dict, dataset_config