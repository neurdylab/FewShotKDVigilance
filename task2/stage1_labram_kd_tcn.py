import random 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader, ConcatDataset
import pytorch_lightning as pl
import numpy as np
import tqdm
import Modules.LaBraM.modeling_finetune
import timm.models
from timm.models import create_model
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from timm.models import create_model
from utils_EEGPT import temporal_interpolation
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from collections import OrderedDict
from vigilance_dataset_1024 import *

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) 
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
            "loss_kd_logit": self.loss_kd_logit,
            "loss_kd_feat": self.loss_kd_feat,
            "loss_averaged_gt": self.loss_averaged_gt,
        }
    
    def loss_kd_logit(self, eeg_feats, fmri_feats, x_eeg, x_fmri, logits_eeg, logits_fmri, target):
        teacher = logits_eeg.detach()
        teacher = (teacher - teacher.mean(dim=-1, keepdim=True)) / (teacher.std(dim=-1, keepdim=True) + 1e-6)
        student = (logits_fmri - logits_fmri.mean(dim=-1, keepdim=True)) / (logits_fmri.std(dim=-1, keepdim=True) + 1e-6)
        return {"loss_kd_logit": F.mse_loss(student, teacher)}

    def loss_kd_feat(self, eeg_feats, fmri_feats, x_eeg, x_fmri, logits_eeg, logits_fmri, target):
        eeg_feats = F.normalize(eeg_feats.detach(), dim=-1)
        fmri_feats = F.normalize(fmri_feats, dim=-1)
        loss = 1.0 - (fmri_feats * eeg_feats).sum(dim=-1).mean()
        return {"loss_kd_feat": loss}
    
    def loss_averaged_gt(self, eeg_feats, fmri_feats, x_eeg, x_fmri, logits_eeg, logits_fmri, target):
        logits_fmri_mean = logits_fmri.mean(dim=-1).reshape(-1) 
        target_mean = target.mean(dim=-1).reshape(-1) 
        return {"loss_averaged_gt": F.mse_loss(logits_fmri_mean, target_mean)}

    def single_output_forward(self, eeg_feats, fmri_feats, x_eeg, x_fmri, logits_eeg, logits_fmri, target):
        losses = {}
        for f in self.loss_functions: 
            loss_wt_key = f + "_weight"
            if (
                loss_wt_key in self.loss_weight_dict
                and self.loss_weight_dict[loss_wt_key] > 0
            ) or loss_wt_key not in self.loss_weight_dict:
                curr_loss = self.loss_functions[f](eeg_feats, fmri_feats, x_eeg, x_fmri, logits_eeg, logits_fmri, target)
                losses.update(curr_loss)
        final_loss = 0.0
        for w in self.loss_weight_dict:
            if self.loss_weight_dict[w] > 0:
                losses[w.replace("_weight", "")] *= self.loss_weight_dict[w]
                final_loss += losses[w.replace("_weight", "")]
        return final_loss, losses
    
    def forward(self, eeg_feats, fmri_feats, x_eeg, x_fmri, logits_eeg, logits_fmri, target):
        loss, loss_dict = self.single_output_forward(eeg_feats, fmri_feats, x_eeg, x_fmri, logits_eeg, logits_fmri, target)
        return loss, loss_dict


def build_criterion():
    loss_weight_dict = {
        "loss_kd_logit": 1.0,
        "loss_kd_feat": 1.0,
        "loss_averaged_gt": 5.0
    }
    criterion = SetCriterion(loss_weight_dict)
    return criterion


class ResidualDilatedConvBlock(nn.Module):
    def __init__(self, dim, dilation=1, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation),
            nn.Dropout(dropout),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))
    
    
class LitEEGPTCausal(pl.LightningModule):
    
    def __init__(self, pretrained_ckpt_path=None, training_base=None):
        super().__init__()    
        self.chans_num = len(use_channels_names)
        
        model = create_model("labram_base_patch200_200", 
                                qkv_bias=False,
                                rel_pos_bias=True,
                                num_classes=2,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                attn_drop_rate=0.0,
                                drop_block_rate=None,
                                use_mean_pooling=True,
                                init_scale=0.001,
                                use_rel_pos_bias=True,
                                use_abs_pos_emb=True,
                                init_values=0.1,)
        if pretrained_ckpt_path is None:
            checkpoint = torch.load("Modules/LaBraM/labram-base.pth")
            new_checkpoint = {}
            for k,v in checkpoint['model'].items():
                if k.startswith('student.'):
                    new_checkpoint[k[len('student.'):]] = v
            model.load_state_dict(new_checkpoint, strict=False)
            
        self.eeg_encoder        = model
        self.eeg_mapper = nn.Sequential(
                          nn.Linear(5200, 2048),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                          nn.Linear(2048, 1024),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                          nn.Linear(1024, 1024),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                          nn.Linear(1024, 512),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                        )
        self.temporal_decoder = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=5, padding=2)
        )
        
        self.fmri_encoder = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1),
                                          nn.GELU(),
                                          ResidualDilatedConvBlock(1024, dilation=1, dropout=0.15),
                                          ResidualDilatedConvBlock(1024, dilation=2, dropout=0.15),
                                          ResidualDilatedConvBlock(1024, dilation=4, dropout=0.15),
                                          ResidualDilatedConvBlock(1024, dilation=8, dropout=0.15),)              
        self.fmri_mapper = nn.Sequential(
                          nn.Linear(1024, 512),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                        )
        self.fmri_temporal_decoder = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )
        
        self.criterion = build_criterion()
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity = True
        if pretrained_ckpt_path is not None:
            self.load_pretrained(pretrained_ckpt_path)
        if training_base is not None:
            self.load_training_base(training_base)
            self._reset_parameters()
    
    def load_pretrained(self, ckpt_path):
        pretrain_ckpt = torch.load(ckpt_path)
        state_dict = pretrain_ckpt['state_dict']
        eeg_encoder_state = OrderedDict({k.replace("eeg_encoder.", ""): v for k, v in state_dict.items() if k.startswith("eeg_encoder.")})
        eeg_mapper_state = OrderedDict({k.replace("eeg_mapper.", ""): v for k, v in state_dict.items() if k.startswith("eeg_mapper.")})
        temporal_decoder_state = OrderedDict({k.replace("temporal_decoder.", ""): v for k, v in state_dict.items() if k.startswith("temporal_decoder.")})

        print("\nLoading eeg_encoder...")
        msg = self.eeg_encoder.load_state_dict(eeg_encoder_state, strict=False)
        print("Missing:", msg.missing_keys)
        print("Unexpected:", msg.unexpected_keys)

        print("\nLoading eeg_mapper...")
        msg = self.eeg_mapper.load_state_dict(eeg_mapper_state, strict=False)
        print("Missing:", msg.missing_keys)
        print("Unexpected:", msg.unexpected_keys)

        print("\nLoading temporal_decoder...")
        msg = self.temporal_decoder.load_state_dict(temporal_decoder_state, strict=False)
        print("Missing:", msg.missing_keys)
        print("Unexpected:", msg.unexpected_keys)
        
        fmri_encoder_state = OrderedDict({k.replace("fmri_encoder.", ""): v for k, v in state_dict.items() if k.startswith("fmri_encoder.")})
        fmri_mapper_state = OrderedDict({k.replace("fmri_mapper.", ""): v for k, v in state_dict.items() if k.startswith("fmri_mapper.")})
        fmri_temporal_decoder_state = OrderedDict({k.replace("fmri_temporal_decoder.", ""): v for k, v in state_dict.items() if k.startswith("fmri_temporal_decoder.")})
        
        print("\nLoading fmri_encoder...")
        msg = self.fmri_encoder.load_state_dict(fmri_encoder_state, strict=False)
        print("Missing:", msg.missing_keys)
        print("Unexpected:", msg.unexpected_keys)

        print("\nLoading fmri_mapper...")
        msg = self.fmri_mapper.load_state_dict(fmri_mapper_state, strict=False)
        print("Missing:", msg.missing_keys)
        print("Unexpected:", msg.unexpected_keys)

        print("\nLoading fmri_temporal_decoder...")
        msg = self.fmri_temporal_decoder.load_state_dict(fmri_temporal_decoder_state, strict=False)
        print("Missing:", msg.missing_keys)
        print("Unexpected:", msg.unexpected_keys)

    def load_training_base(self, ckpt_path):
        pretrain_ckpt = torch.load(ckpt_path)
        state_dict = pretrain_ckpt['state_dict']
        eeg_encoder_state = OrderedDict({k.replace("eeg_encoder.", ""): v for k, v in state_dict.items() if k.startswith("eeg_encoder.")})
        eeg_mapper_state = OrderedDict({k.replace("eeg_mapper.", ""): v for k, v in state_dict.items() if k.startswith("eeg_mapper.")})
        temporal_decoder_state = OrderedDict({k.replace("temporal_decoder.", ""): v for k, v in state_dict.items() if k.startswith("temporal_decoder.")})

        print("\nLoading eeg_encoder...")
        msg = self.eeg_encoder.load_state_dict(eeg_encoder_state, strict=False)
        print("Missing:", msg.missing_keys)
        print("Unexpected:", msg.unexpected_keys)

        print("\nLoading eeg_mapper...")
        msg = self.eeg_mapper.load_state_dict(eeg_mapper_state, strict=False)
        print("Missing:", msg.missing_keys)
        print("Unexpected:", msg.unexpected_keys)

        print("\nLoading temporal_decoder...")
        msg = self.temporal_decoder.load_state_dict(temporal_decoder_state, strict=False)
        print("Missing:", msg.missing_keys)
        print("Unexpected:", msg.unexpected_keys)
        
    def _reset_parameters(self):
        func = nn.init.xavier_uniform_
        for p in self.fmri_encoder.parameters():
            if p.dim() > 1:
                func(p)
        for p in self.fmri_mapper.parameters():
            if p.dim() > 1:
                func(p)
        for p in self.fmri_temporal_decoder.parameters():
            if p.dim() > 1:
                func(p)
    
    def forward(self, eeg, fmri):
        self.eeg_encoder.eval()
        self.eeg_mapper.eval()
        self.temporal_decoder.eval()
        for m in [self.eeg_encoder, self.eeg_mapper, self.temporal_decoder]:
            for p in m.parameters():
                p.requires_grad = False
    
        B, T_unmasked, C = eeg.shape 
        eeg = eeg.permute(0, 2, 1) 
        B, C, T = eeg.shape
        eeg_feats = []
        fmri_feats = []
        fmri = fmri.permute(0, 2, 1) 
        for i in range(T//(525*50)):
            x = temporal_interpolation(eeg[:, :, i*525*50:(i+1)*525*50], 200*50) 
            x = x.reshape((B,C,x.shape[-1]//200,200)) 
            x = x.to(torch.float32)
            feats = self.eeg_encoder.patch_embed(x) 
            feats = self.eeg_encoder.pos_drop(feats) 
            for block in self.eeg_encoder.blocks:
                feats = block(feats)
            feats = self.eeg_encoder.norm(feats)
            feats = self.eeg_encoder.fc_norm(feats)
            feats = feats.reshape(B, C, 50, 200)    
            feats = feats.permute(0, 2, 1, 3)          
            feats = feats.reshape(B, 50, C * 200) 
            eeg_feats.append(feats)
            
            fmri_seg = fmri[:, :, i*50:(i+1)*50] 
            fmri_feat = self.fmri_encoder(fmri_seg)
            fmri_mapped_feat = self.fmri_mapper(fmri_feat.transpose(1, 2)) 
            fmri_feats.append(fmri_mapped_feat)
            
        eeg_feats = torch.cat([feat for feat in eeg_feats], dim=0) 
        eeg_feats = self.eeg_mapper(eeg_feats) 
        x_eeg = eeg_feats.permute(0, 2, 1)  
        logits_eeg = self.temporal_decoder(x_eeg)  
        fmri_feats = torch.cat([feat for feat in fmri_feats], dim=0)
        x_fmri = fmri_feats.permute(0, 2, 1)
        logits_fmri = self.fmri_temporal_decoder(x_fmri) 
        return eeg_feats, fmri_feats, x_eeg, x_fmri, logits_eeg, logits_fmri

    def training_step(self, batch, batch_idx):
        fmri = batch["fmri_data"].permute(0, 2, 1).float() 
        eeg = batch["eeg_data"].permute(0, 2, 1).float() 
        alpha_tot_ratio_smoothed = batch["alpha_tot_ratio_smoothed"]
        eeg_feats, fmri_feats, x_eeg, x_fmri, logits_eeg, logits_fmri = self.forward(eeg, fmri) 
        target = torch.stack([alpha_tot_ratio_smoothed], dim=1).float()  
        final_loss, losses = self.criterion.single_output_forward(eeg_feats, fmri_feats, x_eeg, x_fmri, logits_eeg, logits_fmri, target)

        pred_np = logits_fmri.detach().cpu().numpy() 
        gt_np = target.detach().cpu().numpy()
        
        corrs = []
        for band_idx in range(pred_np.shape[1]):
            corr_band = []
            for i in range(pred_np.shape[0]):
                corr, _ = pearsonr(pred_np[i, band_idx, :], gt_np[i, band_idx, :])
                corr_band.append(corr)
            corrs.append(corr_band)
        corrs = np.array(corrs)
        self.log("train_alpha_corr", np.mean(corrs[0]), on_epoch=True, sync_dist=True)
        for key in losses.keys():
            self.log("train_"+key, losses[key], on_epoch=True, sync_dist=True)
        self.log("train_loss_total", final_loss, on_epoch=True, sync_dist=True)
        return final_loss
        
    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"]=[]
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity = False
            return super().on_validation_epoch_end()
        preds, targets = zip(*self.running_scores["valid"])  
        pred = torch.cat(preds).numpy() 
        target = torch.cat(targets).numpy() 
        corrs = []
        for band_idx in range(pred.shape[1]):
            corr_band = []
            for i in range(pred.shape[0]): 
                corr, _ = pearsonr(pred[i, band_idx, :], target[i, band_idx, :])
                corr_band.append(corr)
            corrs.append(corr_band)
        corrs = np.array(corrs) 
        val_mse_result = mean_squared_error(pred.reshape(-1), target.reshape(-1))
        self.log("val_alpha_corr", np.mean(corrs[0]), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_mse_result", val_mse_result, prog_bar=True, on_epoch=True, sync_dist=True)
        return super().on_validation_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        fmri = batch["fmri_data"].permute(0, 2, 1).float() 
        eeg = batch["eeg_data"].permute(0, 2, 1).float() 
        alpha_tot_ratio_smoothed = batch["alpha_tot_ratio_smoothed"]
        eeg_feats, fmri_feats, x_eeg, x_fmri, logits_eeg, logits_fmri = self.forward(eeg, fmri) 
        target = torch.stack([alpha_tot_ratio_smoothed], dim=1).float()  
        final_loss, losses = self.criterion.single_output_forward(eeg_feats, fmri_feats, x_eeg, x_fmri, logits_eeg, logits_fmri, target)
        for key in losses.keys():
            self.log("valid_"+key, losses[key], on_epoch=True, sync_dist=True)
        self.log("valid_loss_total", final_loss, on_epoch=True, sync_dist=True)
        self.running_scores["valid"].append((logits_fmri.detach().cpu(), target.detach().cpu()))
        return final_loss
    
    def test_step(self, batch, batch_idx):
        fmri = batch["fmri_data"].permute(0, 2, 1).float() 
        eeg = batch["eeg_data"].permute(0, 2, 1).float() 
        alpha_tot_ratio_smoothed = batch["alpha_tot_ratio_smoothed"]
        eeg_feats, fmri_feats, x_eeg, x_fmri, logits_eeg, logits_fmri = self.forward(eeg, fmri) 
        target = torch.stack([alpha_tot_ratio_smoothed], dim=1).float()  
        final_loss, losses = self.criterion.single_output_forward(eeg_feats, fmri_feats, x_eeg, x_fmri, logits_eeg, logits_fmri, target)
        self.running_scores["test"].append((logits_fmri.detach().cpu(), target.detach().cpu()))
        return final_loss

    def on_test_epoch_start(self) -> None:
        self.running_scores["test"] = []
        return super().on_test_epoch_start()

    def on_test_epoch_end(self):
        preds, targets = zip(*self.running_scores["test"]) 
        pred = torch.cat(preds).numpy()
        target = torch.cat(targets).numpy()
        corrs = []
        for band_idx in range(pred.shape[1]):
            corr_band = []
            for i in range(pred.shape[0]): 
                corr, _ = pearsonr(pred[i, band_idx, :], target[i, band_idx, :])
                corr_band.append(corr)
            corrs.append(corr_band)
        corrs = np.array(corrs) 
        test_mse_result = mean_squared_error(pred.reshape(-1), target.reshape(-1))
        self.log("test_alpha_corr", np.mean(corrs[0]), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("test_mse_result", test_mse_result, prog_bar=True, on_epoch=True, sync_dist=True)
        return super().on_test_epoch_end()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.fmri_encoder.parameters())+
            list(self.fmri_mapper.parameters())+
            list(self.fmri_temporal_decoder.parameters()),
            lr=max_lr,
            weight_decay=0.01)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2)
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': 1, 
            'monitor': 'val_loss', 
            'strict': True, 
            'name': None, 
        }
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )

def concatenation_w_ratio(datasets, ratios):
    dataset_list = []
    for i in range(len(datasets)):
        n = int(len(datasets[i])*ratios[i])
        subset = Subset(datasets[i], range(n))
        dataset_list.append(subset)
    final_dataset = ConcatDataset(dataset_list)
    return final_dataset


if __name__=="__main__":
    import math
    global max_epochs
    global steps_per_epoch
    global max_lr
    torch.set_float32_matmul_precision('medium' )
    batch_size=16
    nih_ecr_zeroshot_dataset = VigilanceDataset1024(dataset_name="nih_ecr", window_size=50, step_size=5, vigilance_threshold=-25, split_set="zero_shot")
    nih_ect_zeroshot_dataset = VigilanceDataset1024(dataset_name="nih_ect", window_size=50, step_size=5, vigilance_threshold=-25, split_set="zero_shot")

    vu_healthy_train_dataset = VigilanceDataset1024(dataset_name="vu_healthy", window_size=50, step_size=5, vigilance_threshold=-25, split_set="train")
    vu_healthy_test_dataset = VigilanceDataset1024(dataset_name="vu_healthy", window_size=50, step_size=50, vigilance_threshold=-25, split_set="test")
    
    dataset_list = [vu_healthy_train_dataset, nih_ecr_zeroshot_dataset, nih_ect_zeroshot_dataset]
    ratio_list = [1, 1, 1]
    train_dataset = concatenation_w_ratio(dataset_list, ratio_list)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=8)
    test_loader  = DataLoader(vu_healthy_test_dataset,  batch_size=batch_size, shuffle=False, num_workers=8)
    max_epochs = 50
    steps_per_epoch = math.ceil(len(train_loader))
    max_lr = 3e-4
    folder = "" #TODO
    name = "" #TODO
    ckpt_cb = ModelCheckpoint(
        dirpath=folder + name,
        filename="best-test-mCorr-{epoch:02d}-{val_alpha_corr:.4f}",
        monitor="val_alpha_corr",
        mode="max",
        save_top_k=1,
        verbose=True
    )
    model = LitEEGPTCausal(training_base='') #TODO
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    early_stop_cb = EarlyStopping(
        monitor="val_alpha_corr",
        mode="max",
        patience=15,
        min_delta=1e-4,
        verbose=True
    )
    trainer = pl.Trainer(
        accelerator='cuda',
        precision=32,
        max_epochs=max_epochs,
        callbacks=[lr_monitor, ckpt_cb, early_stop_cb],
        logger=[
            pl_loggers.TensorBoardLogger(folder, name=name, version="single-run"),
            pl_loggers.CSVLogger(folder, name=name, version="single-run"),
        ]
    )
    trainer.fit(
        model,
        train_loader,
        test_loader,
    )
    print("Best‐on‐test checkpoint:", ckpt_cb.best_model_path)