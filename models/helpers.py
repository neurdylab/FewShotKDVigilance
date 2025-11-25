# Copyright (c) Facebook, Inc. and its affiliates.
import torch.nn as nn
import torch
from functools import partial
import copy


class BatchNormDim1Swap(nn.BatchNorm1d):
    """
    Used for nn.Transformer that uses a HW x N x C rep
    """

    def forward(self, x):
        """
        x: HW x N x C
        permute to N x C x HW
        Apply BN on C
        permute back
        """
        hw, n, c = x.shape
        x = x.permute(1, 2, 0)
        x = super(BatchNormDim1Swap, self).forward(x)
        # x: n x c x hw -> hw x n x c
        x = x.permute(2, 0, 1)
        return x


NORM_DICT = {
    "bn": BatchNormDim1Swap,
    "bn1d": nn.BatchNorm1d,
    "id": nn.Identity,
    "ln": nn.LayerNorm,
}

ACTIVATION_DICT = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leakyrelu": partial(nn.LeakyReLU, negative_slope=0.1),
}

WEIGHT_INIT_DICT = {
    "xavier_uniform": nn.init.xavier_uniform_,
}

class GenericMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        norm_fn_name=None,
        activation="relu",
        use_conv=False,
        dropout=None,
        hidden_use_bias=False,
        output_use_bias=True,
        output_use_activation=False,
        output_use_norm=False,
        weight_init_name=None,
    ):
        super().__init__()
        activation = ACTIVATION_DICT[activation]
        norm = None
        if norm_fn_name is not None:
            norm = NORM_DICT[norm_fn_name]
        if norm_fn_name == "ln" and use_conv:
            norm = lambda x: nn.GroupNorm(1, x)  # easier way to use LayerNorm

        if dropout is not None:
            if not isinstance(dropout, list):
                dropout = [dropout for _ in range(len(hidden_dims))]

        layers = []
        prev_dim = input_dim
        for idx, x in enumerate(hidden_dims):
            if use_conv:
                layer = nn.Conv1d(prev_dim, x, 1, bias=hidden_use_bias)
            else:
                layer = nn.Linear(prev_dim, x, bias=hidden_use_bias)
            layers.append(layer)
            if norm:
                layers.append(norm(x))
            layers.append(activation())
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout[idx]))
            prev_dim = x
        if use_conv:
            layer = nn.Conv1d(prev_dim, output_dim, 1, bias=output_use_bias)
        else:
            layer = nn.Linear(prev_dim, output_dim, bias=output_use_bias)
        layers.append(layer)

        if output_use_norm:
            layers.append(norm(output_dim))

        if output_use_activation:
            layers.append(activation())

        self.layers = nn.Sequential(*layers)

        if weight_init_name is not None:
            self.do_weight_init(weight_init_name)

    def do_weight_init(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for (_, param) in self.named_parameters():
            if param.dim() > 1:  # skips batchnorm/layernorm
                func(param)

    def forward(self, x):
        output = self.layers(x)
        return output

class GenericMLP_Pred(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        norm_fn_name=None,
        activation="relu",
        use_conv=False,
        dropout=None,
        hidden_use_bias=False,
        output_use_bias=True,
        output_use_activation=False,
        output_use_norm=False,
        weight_init_name=None,
    ):
        super().__init__()
        activation = ACTIVATION_DICT[activation]
        norm = None
        if norm_fn_name is not None:
            norm = NORM_DICT[norm_fn_name]
        if norm_fn_name == "ln" and use_conv:
            norm = lambda x: nn.GroupNorm(1, x)  # easier way to use LayerNorm

        if dropout is not None:
            if not isinstance(dropout, list):
                dropout = [dropout for _ in range(len(hidden_dims))]

        layers = []
        prev_dim = input_dim
        for idx, x in enumerate(hidden_dims):
            if use_conv:
                layer = nn.Conv1d(prev_dim, x, 1, bias=hidden_use_bias)
            else:
                layer = nn.Linear(prev_dim, x, bias=hidden_use_bias)
            layers.append(layer)
            if norm:
                layers.append(norm(x))
            layers.append(activation())
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout[idx]))
            prev_dim = x
        if use_conv:
            layer = nn.Conv1d(prev_dim, output_dim, 1, bias=output_use_bias)
        else:
            layer = nn.Linear(prev_dim, output_dim, bias=output_use_bias)
        layers.append(layer)

        if output_use_norm:
            layers.append(norm(output_dim))

        if output_use_activation:
            layers.append(activation())

        self.layers = nn.Sequential(*layers)

        if weight_init_name is not None:
            self.do_weight_init(weight_init_name)

    def do_weight_init(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for (_, param) in self.named_parameters():
            if param.dim() > 1:  # skips batchnorm/layernorm
                func(param)

    def forward(self, x):
        output = self.layers(x)

        # Notice: delete this part in classification models!
        output = torch.tanh(output)
        return output
    
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class SequentialCNN2D(nn.Module):
    def __init__(self, in_channels=3, feature_dim=128):
        super(SequentialCNN2D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(127232, feature_dim),  # 54528
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x) # ([456, 128, 3, 142])
        x = self.fc(x) 
        return x

class SequentialCNN2D_Expand(nn.Module):
    def __init__(self, in_channels=3, feature_dim=256):
        super(SequentialCNN2D_Expand, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(127232, feature_dim),  
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)  
        x = self.fc(x)  
        return x