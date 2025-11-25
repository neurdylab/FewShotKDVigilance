# pure EEG modality
import random 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
import pytorch_lightning as pl
from functools import partial
import numpy as np
import random
import os 
import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score
from models.transformer import (TransformerEncoder,
                                TransformerEncoderLayer)
from models.helpers import (ACTIVATION_DICT, NORM_DICT, WEIGHT_INIT_DICT,
                            get_clones)
from models.helpers import GenericMLP
from models.helpers import get_clones, WEIGHT_INIT_DICT, NORM_DICT, ACTIVATION_DICT
import Modules.LaBraM.modeling_finetune
import timm.models
from timm.models import create_model
import torch
from utils_EEGPT import temporal_interpolation
from utils_eval import get_metrics
from Modules.Transformers.pos_embed import create_1d_absolute_sin_cos_embedding
from Modules.models.EEGPT_mcae import EEGTransformer
from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint
from vigilance_datasets import build_dataset, build_dataset_zeroshot, build_dataset_fewshot
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

from collections import OrderedDict

from Modules.BIOT.biot import (
    BIOTClassifier,
)

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch(11)

use_channels_names_original = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8',
                        'P7', 'P8', 'FPZ', 'FZ', 'CZ', 'PZ', 'POZ', 'OZ', 'FT9', 'FT10', 'TP9', 'TP10']
use_channels_names = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8',
                        'P7', 'P8', 'FPZ', 'FZ', 'CZ', 'PZ', 'POZ', 'OZ', 'F7', 'F8', 'T7', 'T8']


class SetCriterion(nn.Module):
    def __init__(self, loss_weight_dict):
        super().__init__()
        self.loss_weight_dict = loss_weight_dict
        self.loss_functions = {
            "loss_eeg_cls": self.loss_eeg_cls,
        }
        
    def loss_eeg_cls(self, eeg_feats, logits, vigilance_seg):
        vigilance_seg = vigilance_seg.view(-1).long()
        ce_loss_fn = nn.CrossEntropyLoss()
        ce_loss = ce_loss_fn(logits, vigilance_seg)
        return {"loss_eeg_cls": ce_loss}

    def single_output_forward(self, eeg_feats, logits, vigilance_seg):
        losses = {}
        for f in self.loss_functions: 
            loss_wt_key = f + "_weight"
            if (
                loss_wt_key in self.loss_weight_dict
                and self.loss_weight_dict[loss_wt_key] > 0
            ) or loss_wt_key not in self.loss_weight_dict:
                curr_loss = self.loss_functions[f](eeg_feats, logits, vigilance_seg)
                losses.update(curr_loss)
        final_loss = 0.0
        for w in self.loss_weight_dict:
            if self.loss_weight_dict[w] > 0:
                losses[w.replace("_weight", "")] *= self.loss_weight_dict[w]
                final_loss += losses[w.replace("_weight", "")]
        return final_loss, losses
    
    def forward(self, eeg_feats, logits, vigilance_seg):
        loss, loss_dict = self.single_output_forward(eeg_feats, logits, vigilance_seg)
        return loss, loss_dict


def build_criterion():
    loss_weight_dict = {
        "loss_eeg_cls": 1.0,
    }
    criterion = SetCriterion(loss_weight_dict)
    return criterion


class LitEEGPTCausal(pl.LightningModule):
    
    def __init__(self, load_path="EEGvigilance/stage1_BIOT_fewshot/best.ckpt"):
        super().__init__()    
        in_channels = 18
        self.chan_conv      = Conv1dWithConstraint(26, in_channels, 1, max_norm=1)
        self.num_class = 2
        model = BIOTClassifier(
                    n_classes=self.num_class,
                    n_channels=in_channels,
                    n_fft=200,
                    hop_length=100,
                )
        self.eeg_encoder        = model
    
        self.eeg_mapper = nn.Sequential(
                          nn.Linear(256, 1024),
                          nn.ReLU(),
                          nn.Dropout(0.2),
                          nn.Linear(1024, 512*5),
                          nn.ReLU(),
                          nn.Dropout(0.2),
                        )
        self.eeg_classfier = nn.Sequential(
                                    nn.Linear(512*5, 256),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(256, 2)
                                )
        
        pretrain_ckpt = torch.load(load_path)
        state_dict = pretrain_ckpt['state_dict']
        eeg_encoder_state = OrderedDict({k.replace("eeg_encoder.", ""): v for k, v in state_dict.items() if k.startswith("eeg_encoder.")})
        eeg_mapper_state = OrderedDict({k.replace("eeg_mapper.", ""): v for k, v in state_dict.items() if k.startswith("eeg_mapper.")})
        eeg_classifier_state = OrderedDict({k.replace("eeg_classfier.", ""): v for k, v in state_dict.items() if k.startswith("eeg_classfier.")})
        chan_conv_state = {k.replace("chan_conv.", ""): v for k, v in state_dict.items() if k.startswith("chan_conv.")}

        self.eeg_encoder.load_state_dict(eeg_encoder_state)
        self.eeg_mapper.load_state_dict(eeg_mapper_state)
        self.eeg_classfier.load_state_dict(eeg_classifier_state)
        self.chan_conv.load_state_dict(chan_conv_state)

        self.criterion = build_criterion()
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity = True
    
    def forward(self, eeg, fmri):
        B, T_unmasked, C = eeg.shape 
        eeg = eeg.permute(0, 2, 1) 
        B, C, T = eeg.shape
        eeg_feats = []
        for i in range(T//(525*5)):
            x = temporal_interpolation(eeg[:, :, i*525*5:(i+1)*525*5], 200*15) # ([32, 26, 3000])
            x = self.chan_conv(x) # ([32, 16, 3000])
            x = x.to(torch.float32)
            self.eeg_encoder.eval()
            h = self.eeg_encoder.biot(x)
            eeg_feats.append(h)
            
        eeg_feats = torch.cat([feat for feat in eeg_feats], dim=0) # ([32, 256])
        eeg_feats = self.eeg_mapper(eeg_feats) 
        eeg_all_feats = eeg_feats.reshape(eeg_feats.shape[0], -1)
        logits = self.eeg_classfier(eeg_all_feats) 
        return eeg_all_feats, logits

    def training_step(self, batch, batch_idx):
        fmri, eeg, physio, eeg_index_linear_raw, eeg_index_linear_smoothed, eeg_index_binary, alpha_theta_ratio, vigilance_seg = batch
        eeg_feats, logits = self.forward(eeg, fmri) 
        final_loss, losses = self.criterion.single_output_forward(eeg_feats, logits, vigilance_seg)
        pred = np.argmax(logits.detach().cpu().numpy(), axis=1)
        gt_flat = vigilance_seg.reshape(-1).detach().cpu().numpy()
        mean_f1 = f1_score(gt_flat, pred, average='macro') 
        for key in losses.keys():
            self.log("train_"+key, losses[key], on_epoch=True, sync_dist=True)
        self.log("train_loss_total", final_loss, on_epoch=True, sync_dist=True)
        self.log("train_mf1", mean_f1, on_epoch=True, sync_dist=True)
        return final_loss
        
    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"]=[]
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity = False
            return super().on_validation_epoch_end()
        gts, logits = zip(*self.running_scores["valid"])  # List of (B, 10)
        gt = torch.cat(gts).numpy()
        logit = torch.cat(logits).numpy()
        pred = np.argmax(logit, axis=1) 
        gt_flat = gt.reshape(-1)
        mean_f1 = f1_score(gt_flat, pred, average='macro') 
        self.log("val_mf1", mean_f1, prog_bar=True, on_epoch=True)
        return super().on_validation_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        fmri, eeg, physio, eeg_index_linear_raw, eeg_index_linear_smoothed, eeg_index_binary, alpha_theta_ratio, vigilance_seg = batch
        eeg_feats, logits = self.forward(eeg, fmri) 
        final_loss, losses = self.criterion.single_output_forward(eeg_feats, logits, vigilance_seg)
        for key in losses.keys():
            self.log("valid_"+key, losses[key], on_epoch=True, sync_dist=True)
        self.log("valid_loss_total", final_loss, on_epoch=True, sync_dist=True)
        self.running_scores["valid"].append((vigilance_seg.detach().cpu(), logits.detach().cpu()))
        return final_loss
    
    def test_step(self, batch, batch_idx):
        fmri, eeg, physio, eeg_index_linear_raw, eeg_index_linear_smoothed, eeg_index_binary, alpha_theta_ratio, vigilance_seg = batch
        eeg_feats, logits = self.forward(eeg, fmri)
        final_loss, losses = self.criterion.single_output_forward(eeg_feats, logits, vigilance_seg)
        self.running_scores["test"].append((vigilance_seg.detach().cpu(), logits.detach().cpu()))
        return final_loss

    def on_test_epoch_start(self) -> None:
        self.running_scores["test"] = []
        return super().on_test_epoch_start()

    def on_test_epoch_end(self):
        gts, logits = zip(*self.running_scores["test"])
        gt = torch.cat(gts).numpy()
        logit = torch.cat(logits).numpy()
        pred = np.argmax(logit, axis=1)
        gt_flat = gt.reshape(-1)
        mean_f1 = f1_score(gt_flat, pred, average='macro')
        self.log("test_mf1", mean_f1, prog_bar=True, on_epoch=True)
        
        from sklearn.metrics import classification_report
        final_report = classification_report(gt_flat, pred, output_dict=True, zero_division=0)
        f1_d = final_report['0']['f1-score']
        f1_a = final_report['1']['f1-score']
        mf1 = final_report['macro avg']['f1-score']

        self.log("test_report_f1_d", f1_d, prog_bar=True, on_epoch=True)
        self.log("test_report_f1_a", f1_a, prog_bar=True, on_epoch=True)
        self.log("test_report_mf1", mf1, prog_bar=True, on_epoch=True)
        print("Zero-shot mF1:", mean_f1)
        return super().on_test_epoch_end()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.eeg_mapper.parameters())+
            list(self.chan_conv.parameters())+
            list(self.eeg_classfier.parameters()),
            weight_decay=0.01)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2)
        lr_dict = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            'interval': 'step',
            'frequency': 1, # The frequency of the scheduler
            'monitor': 'val_loss', # Metric for `ReduceLROnPlateau` to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for `LearningRateMonitor` to use
        }
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )

        
if __name__=="__main__":
    torch.set_float32_matmul_precision('medium' )
    batch_size=32
    nihecr_zeroshot_dataset = build_dataset_zeroshot("NIHECR_alphatheta_smallinterval_1024")
    nihecr_zeroshot_loader  = DataLoader(nihecr_zeroshot_dataset,  batch_size=batch_size, shuffle=False, num_workers=8)
    
    _, eegfmri_vu_test_dataset = build_dataset_fewshot("eegfmri_vu_alphatheta_smallinterval_1024")
    eegfmri_vu_test_loader  = DataLoader(eegfmri_vu_test_dataset,  batch_size=batch_size, shuffle=False, num_workers=8)
    
    trainer = pl.Trainer(accelerator="gpu", precision=16)
    model = LitEEGPTCausal()
    print("testing the EEGfMRIVU dataset few shot performance")
    trainer.test(model, dataloaders=eegfmri_vu_test_loader)
    
    print("\n")
    print("zeroshot of NIH_ECR")
    trainer.test(model, dataloaders=nihecr_zeroshot_loader)
    
    print(f"testing finished")
