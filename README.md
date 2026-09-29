# FewShotKDVigilance
[JMI 2027] Code release for the journal extension

[SPIE 2026] Code release for EEG-to-fMRI knowledge distillation empowers few-shot resting-state fMRI vigilance detection

This codebase is developed based on [<a href="#ref1">1</a>][<a href="#ref2">2</a>][<a href="#ref5">5</a>][<a href="#ref6">6</a>][<a href="#ref7">7</a>]

## Checkpoint
Please download from: https://huggingface.co/alexandraChangLi/FewShotKDVigilance.

## Environment Configs
The environment for data preprocessing and model training can be installed via the following commands:
```bash
conda create --name fewshot_eegpt python=3.8.18
conda activate fewshot_eegpt
conda install numpy
conda install pandas
conda install conda-forge::nibabel
conda install conda-forge::nilearn
conda install conda-forge::mne
conda install pytorch::pytorch
conda install conda-forge::tensorboard
conda install conda-forge::tensorboardx
pip install shap
pip install umap-learn
pip install plotly
pip install matplotlib
pip install torchvision
pip install git+https://github.com/openai/CLIP.git
pip install reformer_pytorch
pip install pytorch-lightning
pip install braindecode  # installs the latest stable release  [oai_citation:0‡PyPI](https://pypi.org/project/braindecode/?utm_source=chatgpt.com)
pip install timm
pip install pyhealth
pip install linear-attention-transformer
conda install -c conda-forge dtaidistance
```
Package versions: \
python: 3.8.18 \
cuda: 12.4 \
numpy: 1.24.3 \
pandas: 1.5.3 \
nibabel: 5.2.1 \
nilearn: 0.10.4 \
mne: 1.6.1 \
pytorch: 2.4.1 \
pytorch-lightning: 2.4.0 \
tensorboard: 2.17.1 \
tensorboardx: 2.6.2.2 \
shap: 0.44.1 \
umap-learn: 0.5.7 \
plotly: 6.2.0 \
matplotlib: 3.7.2 

For detailed versions, please refer to: environment.yaml.

## Dataset Extraction 
EEGfMRI_VU dataset correspond to the training-internal validation dataset, and NIH dataset corresponds to the external validation dataset (ecr and ect). These datasets will be released.

To extract fMRI ROI time series from preprocessed data, run the following:
```bash
cd data_preprocessing/fmri_atlas
python datasetname_fmri_fit_atlas_batch_1024.py
```
We use Matlab for converting the preprocessed EEG data into the .set format (data_preprocessing/eeg_filtering/datasetname_convertEEG_to_set_batch.m) and channel removal (data_preprocessing/eeg_filtering/datasetname_EEG_removechannels_batch.m). Required package: EEGLAB.
Please refer to NeuroBOLT[<a href="#ref3">3</a>] for data preprocessing details and [<a href="#ref4">4</a>] for vigilance ground truth extraction.
For detailed guidance on the dataset preprocessing environment and code, please refer to our prior work[<a href="#ref2">2</a>].

**Note from Authors:** We pick the checkpoint that performs the best on internal validation dataset during training and report the metrics, and we use this selected checkpoint on the unseen external validation dataset. This approached is also applied to baselines compared.

## Model Training and Testing
First we need to download the EEGPT[<a href="#ref1">1</a>]'s codebase, and download the pre-trained EEG foundation models' checkpoints as described in the EEGPT repo.
```bash
git clone https://github.com/BINE022/EEGPT.git
```
Then we move everything from our codebase (/vigilance_datasets/ folder, every .py file for stage 1 and stage 2) to the /EEGPT/downstream/ folder, and delete the original utils.py.
Activate the environment:
```bash
cd EEGPT/downstream/
conda activate fewshot_eegpt
```
## Task 1:
### Stage 1: Vigilance-Guided Latent Space
Note: after training, please rename the best checkpoint to 'best.ckpt'. 
```bash
# EEGPT
nohup python stage1_EEGPT_fewshot_train.py > stage1_EEGPT_fewshot_train.txt & 
nohup python stage1_EEGPT_fewshot_test.py > stage1_EEGPT_fewshot_test.txt &
# LaBraM
nohup python stage1_labram_fewshot_train.py > stage1_labram_fewshot_train.txt &
nohup python stage1_labram_fewshot_test.py > stage1_labram_fewshot_test.txt &
# BIOT
nohup python stage1_BIOT_fewshot_train.py > stage1_BIOT_fewshot_train.txt &
nohup python stage1_BIOT_fewshot_test.py > stage1_BIOT_fewshot_test.txt &
```

### Stage 2: EEG-to-fMRI Knowledge Distillation
Note: after training, please rename the best checkpoint to 'best.ckpt'. 
```bash
# Training
nohup python stage2_labramBIOTEEGPT_kd_transformer_train.py > stage2_labramBIOTEEGPT_kd_transformer_train.txt &
# Evaluation on internal validation set and external validation set
nohup python stage2_labramBIOTEEGPT_kd_transformer_test.py > stage2_labramBIOTEEGPT_kd_transformer_test.txt &
```
Tensorboard can be used for visualization.

### Visualizations
To produce the visualization figures, first run these commands to save the model predictions:
```bash
# Healthy controls
nohup python stage2_labramBIOTEEGPT_kd_transformer_test_visualizations.py > stage2_labramBIOTEEGPT_kd_transformer_test_visualizations.txt & 
# Epilepsy patients
nohup python stage2_labramBIOTEEGPT_kd_transformer_test_visualizations_vpat_gt.py > stage2_labramBIOTEEGPT_kd_transformer_test_visualizations_vpat_gt.txt & 
```
The predictions will be saved at checkpointdirectory/datasetname. Then, go to visualization_umaps.ipynb. Use the same environment for running the notebook.

## Task 2:
### Training for both stages
```bash
nohup python stage0_labram_train.py > stage0_labram_train.txt &

nohup python stage1_labram_kd_tcn.py > stage1_labram_kd_tcn.txt &
```
### Testing
```bash
stage1_labram_kd_tcn_visualizations.ipynb
```

## Potential Questions
Please reach out to chang.li@vanderbilt.edu.

## References
<a id="ref1"></a>[[1] Wang G, Liu W, He Y, et al. Eegpt: Pretrained transformer for universal and reliable representation of eeg signals[J]. Advances in Neural Information Processing Systems, 2024, 37: 39249-39280.](https://neurips.cc/virtual/2024/poster/93793)

<a id="ref2"></a>[[2] Li, C., Li, Y., Pourmotabbed, H., Zhang, S., Salas, J. A., Goodale, S. E., ... & Chang, C. (2025, September). CBrain: Cross-Modal Learning for Brain Vigilance Detection in Resting-State fMRI. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 109-119). Cham: Springer Nature Switzerland.](https://link.springer.com/chapter/10.1007/978-3-032-04927-8_11)

<a id="ref3"></a>[[3] Li Y, Lou A, Xu Z, et al. NeuroBOLT: Resting-state EEG-to-fMRI synthesis with multi-dimensional feature mapping[J]. Advances in neural information processing systems, 2024, 37: 23378-23405.](https://arxiv.org/abs/2410.05341)

<a id="ref4"></a>[[4] Pourmotabbed H, Martin C G, Goodale S E, et al. Multimodal state-dependent connectivity analysis of arousal and autonomic centers in the brainstem and basal forebrain[J]. Imaging Neuroscience, 2025, 3: IMAG. a. 91.](https://direct.mit.edu/imag/article/doi/10.1162/IMAG.a.91/131628)

<a id="ref5"></a>[[5] Misra I, Girdhar R, Joulin A. An end-to-end transformer model for 3d object detection[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2021: 2906-2917.](https://openaccess.thecvf.com/content/ICCV2021/papers/Misra_An_End-to-End_Transformer_Model_for_3D_Object_Detection_ICCV_2021_paper.pdf)

<a id="ref6"></a>[[6] Lu Y, Xu C, Wei X, et al. Open-vocabulary point-cloud object detection without 3d annotation[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023: 1190-1199.](https://openaccess.thecvf.com/content/CVPR2023/papers/Lu_Open-Vocabulary_Point-Cloud_Object_Detection_Without_3D_Annotation_CVPR_2023_paper.pdf)

<a id="ref7"></a>[[7] Gao P, Geng S, Zhang R, et al. Clip-adapter: Better vision-language models with feature adapters[J]. International Journal of Computer Vision, 2024, 132(2): 581-595.](https://link.springer.com/article/10.1007/s11263-023-01891-x)

