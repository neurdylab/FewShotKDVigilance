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
        self.loss_weight_dict = loss_weight_dict.copy()
        self._initial_loss_weight_dict = loss_weight_dict.copy()
        self.loss_functions = {
            "loss_eeg_cls": self.loss_eeg_cls,
            "loss_fmri_cls": self.loss_fmri_cls,
            "loss_kd_logit": self.loss_kd_logit,
            "loss_kd_feat": self.loss_kd_feat,
            "loss_fmri_contrastive": self.loss_fmri_contrastive,
        }
        self.temperature = 2.0
        self.use_pseudo_labels = True
    
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
        
    def loss_eeg_cls(self, eeg_all_feats, eeg_logits, eeg_all_feats_labram, eeg_logits_labram, eeg_all_feats_biot, eeg_logits_biot, fmri_all_feats, fmri_logits, vigilance_seg):
        vigilance_seg = vigilance_seg.view(-1).long()
        ce_loss_fn = nn.CrossEntropyLoss()
        ce_loss = ce_loss_fn(eeg_logits, vigilance_seg)
        return {"loss_eeg_cls": ce_loss}
    
    # def loss_fmri_cls(self, eeg_all_feats, eeg_logits, eeg_all_feats_labram, eeg_logits_labram, fmri_all_feats, fmri_logits, vigilance_seg):
    #     vigilance_seg = vigilance_seg.view(-1).long()
    #     ce_loss_fn = nn.CrossEntropyLoss()
    #     ce_loss = ce_loss_fn(fmri_logits, vigilance_seg)
    #     return {"loss_fmri_cls": ce_loss}
    
    def loss_fmri_cls(self, eeg_all_feats, eeg_logits, eeg_all_feats_labram, eeg_logits_labram, eeg_all_feats_biot, eeg_logits_biot, fmri_all_feats, fmri_logits, vigilance_seg):
        if self.use_pseudo_labels == True:
            print("Use pseudo_labels")
            eeg_logits_combined = 0.5 * eeg_logits + 0.5 * eeg_logits_labram
            pseudo_labels = torch.argmax(eeg_logits_combined.detach(), dim=-1)
            loss = F.cross_entropy(fmri_logits, pseudo_labels)
            return {"loss_fmri_cls": loss}
        else:
            vigilance_seg = vigilance_seg.view(-1).long()
            ce_loss_fn = nn.CrossEntropyLoss()
            return {"loss_fmri_cls": ce_loss_fn(fmri_logits, vigilance_seg)}
        
    def loss_fmri_contrastive(self, eeg_all_feats, eeg_logits, eeg_all_feats_labram, eeg_logits_labram, eeg_all_feats_biot, eeg_logits_biot, fmri_all_feats, fmri_logits, vigilance_seg):
        fmri_contrastive = self.contrastive_loss(fmri_all_feats, vigilance_seg)
        return {"loss_fmri_contrastive": fmri_contrastive}
    
    def loss_kd_logit(self, eeg_all_feats, eeg_logits, eeg_all_feats_labram, eeg_logits_labram, eeg_all_feats_biot, eeg_logits_biot, fmri_all_feats, fmri_logits, vigilance_seg):
        T = self.temperature
        eeg_logits_combined = (eeg_logits + eeg_logits_labram + eeg_logits_biot) / 3
        student_soft = F.log_softmax(fmri_logits / T, dim=-1)
        teacher_soft = F.softmax(eeg_logits_combined / T, dim=-1)
        kd_logit_loss =  F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T ** 2)
        return {"loss_kd_logit": kd_logit_loss}
    
    def loss_kd_feat(self, eeg_all_feats, eeg_logits, eeg_all_feats_labram, eeg_logits_labram, eeg_all_feats_biot, eeg_logits_biot, fmri_all_feats, fmri_logits, vigilance_seg):
        # Teacher A (raw features): EEGPT or BIOT
        loss_a = F.mse_loss(fmri_all_feats, eeg_all_feats)

        loss_b = F.mse_loss(fmri_all_feats, eeg_all_feats_labram)
        # Teacher B (requires normalization): LaBraM
        loss_c = F.mse_loss(
            F.normalize(fmri_all_feats, dim=-1),
            F.normalize(eeg_all_feats_biot, dim=-1)
        )

        # Combine with weights (can be tuned or set equal)

        total_loss = (loss_a + loss_b + loss_c) / 3
        return {"loss_kd_feat": total_loss}

    def single_output_forward(self, eeg_all_feats, eeg_logits, eeg_all_feats_labram, eeg_logits_labram, eeg_all_feats_biot, eeg_logits_biot, fmri_all_feats, fmri_logits, vigilance_seg):
        losses = {}
        for f in self.loss_functions: 
            loss_wt_key = f + "_weight"
            if (
                loss_wt_key in self.loss_weight_dict
                and self.loss_weight_dict[loss_wt_key] > 0
            ) or loss_wt_key not in self.loss_weight_dict:
                curr_loss = self.loss_functions[f](eeg_all_feats, eeg_logits, eeg_all_feats_labram, eeg_logits_labram, eeg_all_feats_biot, eeg_logits_biot, fmri_all_feats, fmri_logits, vigilance_seg)
                losses.update(curr_loss)
        final_loss = 0.0
        for w in self.loss_weight_dict:
            if self.loss_weight_dict[w] > 0:
                losses[w.replace("_weight", "")] *= self.loss_weight_dict[w]
                final_loss += losses[w.replace("_weight", "")]
        return final_loss, losses
    
    def forward(self, eeg_all_feats, eeg_logits, eeg_all_feats_labram, eeg_logits_labram, eeg_all_feats_biot, eeg_logits_biot, fmri_all_feats, fmri_logits, vigilance_seg):
        loss, loss_dict = self.single_output_forward(eeg_all_feats, eeg_logits, eeg_all_feats_labram, eeg_logits_labram, eeg_all_feats_biot, eeg_logits_biot, fmri_all_feats, fmri_logits, vigilance_seg)
        return loss, loss_dict
    
    def update_stage(self, stage: int):
        if stage == 1:
            print("update_stage setting weights in stage 1")
            self.loss_weight_dict.update({
                "loss_kd_logit": 0.4,
                "loss_kd_feat": 0.4,
                "loss_fmri_cls": 0.2,
            })
            self.use_pseudo_labels = True
        elif stage == 2:
            print("update_stage setting weights in stage 2")
            self.loss_weight_dict.update({
                "loss_fmri_cls": 0.8,
                "loss_kd_logit": 0.1,
                "loss_kd_feat": 0.1,
            })
            self.use_pseudo_labels = False


def build_criterion():
    loss_weight_dict = {
        "loss_fmri_cls": 0.7,
        "loss_kd_logit": 0.1,
        "loss_kd_feat": 0.2,
        "loss_fmri_contrastive": 0.1,
    }
    criterion = SetCriterion(loss_weight_dict)
    return criterion


from Modules.BIOT.biot import (
    BIOTClassifier,
)

class LitEEGPTCausal(pl.LightningModule):
    
    def __init__(self, load_eegpt_path="EEGvigilance/stage0_EEGPT_fewshot/best.ckpt", load_labram_path="EEGvigilance/stage0_labram_fewshot/best.ckpt", load_biot_path="EEGvigilance/stage0_BIOT_fewshot/best.ckpt"):
        super().__init__()    
        
        self.chans_num = len(use_channels_names)
        target_encoder = EEGTransformer(
            img_size=[self.chans_num, 256*30],
            patch_size=32*2,
            # patch_stride = 32,
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
    
        self.eeg_encoder = target_encoder
        self.chans_id       = target_encoder.prepare_chan_ids(use_channels_names)
        self.chan_conv       = Conv1dWithConstraint(26, self.chans_num, 1, max_norm=1)
        self.cls_token =        torch.nn.Parameter(torch.rand(1,49152)*0.001, requires_grad=True)
        self.eeg_mapper = nn.Sequential(
                          nn.Linear(98304, 1024),
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
        
        pretrain_ckpt = torch.load(load_eegpt_path)
        state_dict = pretrain_ckpt['state_dict']
        eeg_encoder_state = OrderedDict({k.replace("eeg_encoder.", ""): v for k, v in state_dict.items() if k.startswith("eeg_encoder.")})
        eeg_mapper_state = OrderedDict({k.replace("eeg_mapper.", ""): v for k, v in state_dict.items() if k.startswith("eeg_mapper.")})
        eeg_classifier_state = OrderedDict({k.replace("eeg_classfier.", ""): v for k, v in state_dict.items() if k.startswith("eeg_classfier.")})
        chan_conv_state = {k.replace("chan_conv.", ""): v for k, v in state_dict.items() if k.startswith("chan_conv.")}

        self.eeg_encoder.load_state_dict(eeg_encoder_state)
        self.eeg_mapper.load_state_dict(eeg_mapper_state)
        self.eeg_classfier.load_state_dict(eeg_classifier_state)
        self.chan_conv.load_state_dict(chan_conv_state)
        self.cls_token = torch.nn.Parameter(state_dict['cls_token'], requires_grad=False)
          
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
        self.eeg_encoder_labram        = model
        self.eeg_mapper_labram = nn.Sequential(
                          nn.Linear(15600, 1024),
                          nn.ReLU(),
                          nn.Dropout(0.2),
                          nn.Linear(1024, 512),
                          nn.ReLU(),
                          nn.Dropout(0.2),
                        )
        self.eeg_classfier_labram = nn.Sequential(
                                    nn.Linear(512*5, 256),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(256, 2)
                                )

        pretrain_labram_ckpt = torch.load(load_labram_path)
        state_dict_labram = pretrain_labram_ckpt['state_dict']
        eeg_encoder_state_labram = OrderedDict({k.replace("eeg_encoder.", ""): v for k, v in state_dict_labram.items() if k.startswith("eeg_encoder.")})
        eeg_mapper_state_labram = OrderedDict({k.replace("eeg_mapper.", ""): v for k, v in state_dict_labram.items() if k.startswith("eeg_mapper.")})
        eeg_classifier_state_labram = OrderedDict({k.replace("eeg_classfier.", ""): v for k, v in state_dict_labram.items() if k.startswith("eeg_classfier.")})
        self.eeg_encoder_labram.load_state_dict(eeg_encoder_state_labram)
        self.eeg_mapper_labram.load_state_dict(eeg_mapper_state_labram)
        self.eeg_classfier_labram.load_state_dict(eeg_classifier_state_labram)
        
        in_channels = 18
        self.chan_conv_biot      = Conv1dWithConstraint(26, in_channels, 1, max_norm=1)
        self.num_class = 2
        model_biot = BIOTClassifier(
                    n_classes=self.num_class,
                    n_channels=in_channels,
                    n_fft=200,
                    hop_length=100,
                )
        self.eeg_encoder_biot        = model_biot
    
        self.eeg_mapper_biot = nn.Sequential(
                          nn.Linear(256, 1024),
                          nn.ReLU(),
                          nn.Dropout(0.2),
                          nn.Linear(1024, 512*5),
                          nn.ReLU(),
                          nn.Dropout(0.2),
                        )
        self.eeg_classfier_biot = nn.Sequential(
                                    nn.Linear(512*5, 256),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(256, 2)
                                )

        pretrain_biot_ckpt = torch.load(load_biot_path)
        state_dict_biot = pretrain_biot_ckpt['state_dict']
        eeg_encoder_state_biot = OrderedDict({k.replace("eeg_encoder.", ""): v for k, v in state_dict_biot.items() if k.startswith("eeg_encoder.")})
        eeg_mapper_state_biot = OrderedDict({k.replace("eeg_mapper.", ""): v for k, v in state_dict_biot.items() if k.startswith("eeg_mapper.")})
        eeg_classifier_state_biot= OrderedDict({k.replace("eeg_classfier.", ""): v for k, v in state_dict_biot.items() if k.startswith("eeg_classfier.")})
        chan_conv_state_biot = OrderedDict({k.replace("chan_conv.", ""): v for k, v in state_dict_biot.items() if k.startswith("chan_conv.")})
        self.eeg_encoder_biot.load_state_dict(eeg_encoder_state_biot)
        self.eeg_mapper_biot.load_state_dict(eeg_mapper_state_biot)
        self.eeg_classfier_biot.load_state_dict(eeg_classifier_state_biot)
        self.chan_conv_biot.load_state_dict(chan_conv_state_biot)
          
        self.fmri_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=1024, nhead=4, dim_feedforward=512),
            num_layers=2
        )
        self.fmri_mapper = nn.Sequential(
                          nn.Linear(1024, 1024),
                          nn.ReLU(),
                          nn.Dropout(0.2),
                          nn.Linear(1024, 512),
                          nn.ReLU(),
                          nn.Dropout(0.2),
                        )
        
        self.fmri_classifier = nn.Sequential(
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
        for p in self.fmri_encoder.parameters():
            if p.dim() > 1:
                func(p)
        for p in self.fmri_mapper.parameters():
            if p.dim() > 1:
                func(p)
        for p in self.fmri_classifier.parameters():
            if p.dim() > 1:
                func(p)
                
    def forward(self, eeg, fmri):
        B, T_unmasked, C = eeg.shape 
        eeg = eeg.permute(0, 2, 1) 
        B, C, T = eeg.shape
        eeg_feats = []
        eeg_feats_labram = []
        eeg_feats_biot = []
        fmri_feats = []
        fmri = fmri.permute(0, 2, 1) # ([32, 66, 5])
        
        for i in range(T//(525*5)):
            x = temporal_interpolation(eeg[:, :, i*525*5:(i+1)*525*5], 256*30) # [32, 26, 2625] -> [32, 26, 7680]
            x = x.to(torch.float32)
            x = self.chan_conv(x) # ([32, 26, 7680])
            self.eeg_encoder.eval()
            z = self.eeg_encoder(x, self.chans_id.to(x)) 
            h = z.flatten(2) # ([32, 120, 2048])
            pos = create_1d_absolute_sin_cos_embedding(h.shape[1], dim=2048) # ([120, 2048])
            h = h + pos.repeat((h.shape[0], 1, 1)).to(h) 
            
            B, _, _ = h.shape 
            cls_token_generated = self.cls_token.repeat((5, 1)).to(h.device) 
            h_reshape = h.reshape(B, 5, -1)
            h_with_cls = [torch.cat([cls_token_generated, chunk], dim=1) for chunk in h_reshape] 
            h = torch.cat(h_with_cls, dim=0)
            h = h.reshape(B, 5, -1) # ([32, 5, 98304])
            eeg_feats.append(h)
            
            x_labram = temporal_interpolation(eeg[:, :, i*525*5:(i+1)*525*5], 200*15) # ([32, 26, 3000])
            x_labram = x_labram.reshape((B,C,x_labram.shape[-1]//200,200)) # ([32, 26, 15, 200])
            x_labram = x_labram.to(torch.float32)
            self.eeg_encoder_labram.eval()
            feats = self.eeg_encoder_labram.patch_embed(x_labram)
            feats = self.eeg_encoder_labram.pos_drop(feats)
            for block in self.eeg_encoder_labram.blocks:
                feats = block(feats)
            feats = self.eeg_encoder_labram.norm(feats)
            feats = self.eeg_encoder_labram.fc_norm(feats) # ([32, 390, 200])
            eeg_feats_labram.append(feats)
            
            x_biot = temporal_interpolation(eeg[:, :, i*525*5:(i+1)*525*5], 200*15) # ([32, 26, 3000])
            x_biot = self.chan_conv_biot(x_biot) # ([32, 16, 3000])
            x_biot = x_biot.to(torch.float32)
            self.eeg_encoder_biot.eval()
            h_biot = self.eeg_encoder_biot.biot(x_biot)
            eeg_feats_biot.append(h_biot)
   
            fmri_seg = fmri[:, :, i*5:(i+1)*5] # ([32, 66, 5])
            _, fmri_feat, _ = self.fmri_encoder(fmri_seg.transpose(1, 2)) # ([32, 5, 66])
            fmri_mapped_feat = self.fmri_mapper(fmri_feat) # ([32, 5, 512])
            fmri_feats.append(fmri_mapped_feat)
        
        eeg_feats = torch.cat([feat for feat in eeg_feats], dim=0) # ([32, 5, 98304])
        eeg_feats = self.eeg_mapper(eeg_feats) 
        eeg_all_feats = eeg_feats.reshape(eeg_feats.shape[0], -1)
        eeg_logits = self.eeg_classfier(eeg_all_feats) # ([32, 2])
        
        eeg_feats_labram = torch.cat([feat for feat in eeg_feats_labram], dim=0) # ([32, 5, 98304])
        eeg_feats_labram = eeg_feats_labram.reshape(eeg_feats_labram.shape[0], 5, -1) 
        eeg_feats_labram = self.eeg_mapper_labram(eeg_feats_labram)
        eeg_all_feats_labram = eeg_feats_labram.reshape(eeg_feats_labram.shape[0], -1)
        eeg_logits_labram = self.eeg_classfier_labram(eeg_all_feats_labram) # ([32, 2])
        
        eeg_feats_biot = torch.cat([feat for feat in eeg_feats_biot], dim=0) # ([32, 5, 98304])
        eeg_feats_biot = self.eeg_mapper_biot(eeg_feats_biot)
        eeg_all_feats_biot = eeg_feats_biot.reshape(eeg_feats_biot.shape[0], -1)
        eeg_logits_biot = self.eeg_classfier_biot(eeg_all_feats_biot) # ([32, 2])

        fmri_feats = torch.cat([feat for feat in fmri_feats], dim=0) 
        fmri_all_feats = fmri_feats.reshape(fmri_feats.shape[0], -1)
        fmri_logits = self.fmri_classifier(fmri_all_feats)
        
        return eeg_all_feats, eeg_logits, eeg_all_feats_labram, eeg_logits_labram, eeg_all_feats_biot, eeg_logits_biot, fmri_all_feats, fmri_logits

    def training_step(self, batch, batch_idx):
        
        if self.current_epoch < 20:
            self.criterion.update_stage(1)  # Stage 1: KD only
        else:
            self.criterion.update_stage(2)  # Stage 2: KD + supervised loss
            
        fmri, eeg, physio, eeg_index_linear_raw, eeg_index_linear_smoothed, eeg_index_binary, alpha_theta_ratio, vigilance_seg = batch
        eeg_all_feats, eeg_logits, eeg_all_feats_labram, eeg_logits_labram, eeg_all_feats_biot, eeg_logits_biot, fmri_all_feats, fmri_logits = self.forward(eeg, fmri) 
        final_loss, losses = self.criterion.single_output_forward(eeg_all_feats, eeg_logits, eeg_all_feats_labram, eeg_logits_labram, eeg_all_feats_biot, eeg_logits_biot, fmri_all_feats, fmri_logits, vigilance_seg)
        
        eeg_pred = np.argmax(eeg_logits.detach().cpu().numpy(), axis=1) 
        fmri_pred = np.argmax(fmri_logits.detach().cpu().numpy(), axis=1)
        gt_flat = vigilance_seg.reshape(-1).detach().cpu().numpy()
        eeg_mf1 = f1_score(gt_flat, eeg_pred, average='macro') 
        fmri_mf1 = f1_score(gt_flat, fmri_pred, average='macro') 
        self.log("train_eeg_mf1", eeg_mf1, prog_bar=True, on_epoch=True)
        self.log("train_fmri_mf1", fmri_mf1, prog_bar=True, on_epoch=True)

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
        gts, eeg_logits, fmri_logits = zip(*self.running_scores["valid"])  # List of (B, 10)
        gt = torch.cat(gts).numpy()
        eeg_logit = torch.cat(eeg_logits).numpy()
        fmri_logit = torch.cat(fmri_logits).numpy()
        eeg_pred = np.argmax(eeg_logit, axis=1) 
        fmri_pred = np.argmax(fmri_logit, axis=1)
        gt_flat = gt.reshape(-1)
        eeg_mf1 = f1_score(gt_flat, eeg_pred, average='macro') 
        fmri_mf1 = f1_score(gt_flat, fmri_pred, average='macro') 
        self.log("val_eeg_mf1", eeg_mf1, prog_bar=True, on_epoch=True)
        self.log("val_fmri_mf1", fmri_mf1, prog_bar=True, on_epoch=True)
        return super().on_validation_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        fmri, eeg, physio, eeg_index_linear_raw, eeg_index_linear_smoothed, eeg_index_binary, alpha_theta_ratio, vigilance_seg = batch
        eeg_all_feats, eeg_logits, eeg_all_feats_labram, eeg_logits_labram, eeg_all_feats_biot, eeg_logits_biot, fmri_all_feats, fmri_logits = self.forward(eeg, fmri) 
        final_loss, losses = self.criterion.single_output_forward(eeg_all_feats, eeg_logits, eeg_all_feats_labram, eeg_logits_labram, eeg_all_feats_biot, eeg_logits_biot, fmri_all_feats, fmri_logits, vigilance_seg)
        for key in losses.keys():
            self.log("valid_"+key, losses[key], on_epoch=True, sync_dist=True)
        self.log("valid_loss_total", final_loss, on_epoch=True, sync_dist=True)
        self.running_scores["valid"].append((vigilance_seg.detach().cpu(), eeg_logits.detach().cpu(), fmri_logits.detach().cpu()))
        return final_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.fmri_encoder.parameters())+
            list(self.fmri_mapper.parameters())+
            list(self.fmri_classifier.parameters()),
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
    max_epochs = 30
    steps_per_epoch = math.ceil(len(train_loader))
    max_lr = 7e-4
    folder = "EEGvigilance/"
    name = "stage2_labramBIOTEEGPT_kd_fewshot_transformer"
    ckpt_cb = ModelCheckpoint(
        dirpath=folder + name,
        filename="best-test-mf1-{epoch:02d}-{val_fmri_mf1:.4f}",
        monitor="val_fmri_mf1",
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