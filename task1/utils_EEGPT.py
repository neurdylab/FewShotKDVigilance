import torch
import os
import sys
import numpy as np
import random
import scipy.io as sio
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
from einops import rearrange
import copy
import gc

import scipy
from Data_process.utils import EA

from torch.utils.data import Dataset,DataLoader

from scipy import stats

# from Modules.spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified,SPDVectorize
from Data_process.process_function import Load_BCIC_2a_raw_data
from collections import Counter

def temporal_interpolation(x, desired_sequence_length, mode='nearest', use_avg=True):
    # print(x.shape)
    # squeeze and unsqueeze because these are done before batching
    if use_avg:
        x = x - torch.mean(x, dim=-2, keepdim=True)
    if len(x.shape) == 2:
        return torch.nn.functional.interpolate(x.unsqueeze(0), desired_sequence_length, mode=mode).squeeze(0)
    # Supports batch dimension
    elif len(x.shape) == 3:
        return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
    else:
        raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")
