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

def cal_sim(feat_i, feat_j, temperature):
    feat_i = feat_i / feat_i.norm(dim=len(feat_i.shape)-1, keepdim=True)
    feat_j = feat_j / feat_j.norm(dim=len(feat_j.shape)-1, keepdim=True)
    return feat_i @ feat_j.t() / temperature

class SetCriterion(nn.Module):
    def __init__(self, loss_weight_dict):
        super().__init__()
        self.loss_weight_dict = loss_weight_dict
        self.loss_functions = {
            "loss_eeg_cls": self.loss_eeg_cls,
        }
    
    def contrastive_loss(self, objs_feats, objs_labels, temperature=0.1):
        device = torch.device("cuda" if torch.cuda.is_available else "cpu")
        loss_criterion = nn.CrossEntropyLoss(reduction='mean')
        total_loss = torch.tensor(0, dtype=float).to(device)
        valid_obj_cnt = 1
        for obj_idx in range(objs_feats.shape[0]):
            obj_feature = objs_feats[obj_idx].unsqueeze(0)
            obj_label = objs_labels[obj_idx]
            neg_obj_idxs = torch.where(objs_labels != obj_label)[0]
            neg_obj_idxs = [i for i in neg_obj_idxs]
            if len(neg_obj_idxs) > 0:
                neg_objs = objs_feats[neg_obj_idxs, :]
                neg_loss = cal_sim(obj_feature, neg_objs, temperature)
            else:
                continue
            pos_objs_idxs = torch.where(objs_labels == obj_label)[0]
            pos_objs_idxs = [i for i in pos_objs_idxs if i != obj_idx]

            if len(pos_objs_idxs) > 0:
                pos_objs = objs_feats[pos_objs_idxs, :]
                pos_loss = cal_sim(obj_feature, pos_objs, temperature).t()
            else:
                pos_loss = torch.full((1, 1), 1/temperature)
                valid_obj_cnt -= 1
            pos_loss = pos_loss.to(device)
            neg_loss = neg_loss.to(device)
            logits = torch.cat([pos_loss, neg_loss.repeat(pos_loss.shape[0],1)],dim=1).to(device)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
            
            curr_loss = loss_criterion(logits, labels)
            total_loss += curr_loss
            valid_obj_cnt += 1
        total_loss /= valid_obj_cnt
        return total_loss
    
    def loss_eeg_contrastive(self, eeg_feats, logits, vigilance_seg):
        eeg_contrastive = self.contrastive_loss(eeg_feats, vigilance_seg)
        return {"loss_eeg_contrastive": eeg_contrastive}
    
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
    
    def __init__(self):
        super().__init__()    
        self.num_class = 2
        # init model
        checkpoint = torch.load("Modules/LaBraM/labram-base.pth")
        new_checkpoint = {}
        for k,v in checkpoint['model'].items():
            if k.startswith('student.'):
                new_checkpoint[k[len('student.'):]] = v
        model = create_model("labram_base_patch200_200", 
                                # checkpoint_path= ,
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
        model.load_state_dict(new_checkpoint, strict=False)
        self.eeg_encoder        = model

        self.eeg_mapper = nn.Sequential(
                          nn.Linear(15600, 1024),
                          nn.ReLU(),
                          nn.Dropout(0.2),
                          nn.Linear(1024, 512),
                          nn.ReLU(),
                          nn.Dropout(0.2),
                        )
        self.eeg_classfier = nn.Sequential(
                                    nn.Linear(512*5, 256),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(256, 2)
                                )

        self.criterion = build_criterion()
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity = True
        self._reset_parameters()
    
    
    def _reset_parameters(self):
        func = WEIGHT_INIT_DICT["xavier_uniform"]
        for p in self.eeg_mapper.parameters():
            if p.dim() > 1:
                func(p)
        for p in self.eeg_classfier.parameters():
            if p.dim() > 1:
                func(p)
    
    
    def forward(self, eeg, fmri):
        B, T_unmasked, C = eeg.shape 
        eeg = eeg.permute(0, 2, 1) 
        B, C, T = eeg.shape
        eeg_feats = []
        for i in range(T//(525*5)):
            x = temporal_interpolation(eeg[:, :, i*525*5:(i+1)*525*5], 200*15) # ([32, 26, 3000])
            x = x.reshape((B,C,x.shape[-1]//200,200)) # ([32, 26, 15, 200])
            x = x.to(torch.float32)
            self.eeg_encoder.eval()
            feats = self.eeg_encoder.patch_embed(x)
            feats = self.eeg_encoder.pos_drop(feats)
            for block in self.eeg_encoder.blocks:
                feats = block(feats)
            feats = self.eeg_encoder.norm(feats)
            feats = self.eeg_encoder.fc_norm(feats) # ([32, 390, 200])
            eeg_feats.append(feats)
            
        eeg_feats = torch.cat([feat for feat in eeg_feats], dim=0) # ([32, 390, 200])
        eeg_feats = eeg_feats.reshape(eeg_feats.shape[0], 5, -1) 
        eeg_feats = self.eeg_mapper(eeg_feats) 
        eeg_all_feats = eeg_feats.reshape(eeg_feats.shape[0], -1)
        logits = self.eeg_classfier(eeg_all_feats) # ([32, 2])
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
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.eeg_mapper.parameters())+
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
    import math
    global max_epochs
    global steps_per_epoch
    global max_lr
    torch.set_float32_matmul_precision('medium' )
    batch_size=32
    train_dataset, test_dataset = build_dataset_fewshot("eegfmri_vu_alphatheta_smallinterval_1024")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=8)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=8)
    max_epochs = 10
    steps_per_epoch = math.ceil(len(train_loader))
    max_lr = 3e-5
    folder = "EEGvigilance/"
    name = "stage1_labram_fewshot"
    ckpt_cb = ModelCheckpoint(
        dirpath=folder + name,
        filename="best-test-mf1-{epoch:02d}-{val_mf1:.4f}",
        monitor="val_mf1",
        mode="max",
        save_top_k=1,
        verbose=True
    )
    model = LitEEGPTCausal()
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(
        accelerator='cuda',
        precision=16,
        max_epochs=max_epochs,
        callbacks=[lr_monitor, ckpt_cb],
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
