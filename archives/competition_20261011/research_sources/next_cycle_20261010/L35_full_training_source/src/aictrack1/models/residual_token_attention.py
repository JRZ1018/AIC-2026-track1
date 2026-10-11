"""Original residual readouts on projected tokens; never alter CLIP attention.

The mean-channel control and spatial-attention candidate each add D+1 parameters.
Neither accepts external class names, another encoder, or pretrained readout weights.
Not connected to the production trainer; experimental CPU implementation only.
"""
import math

import torch
from torch import nn


class ResidualTokenReadout(nn.Module):
    def __init__(self, dimension=512, *, mode='spatial', residual_cap=.10):
        super().__init__()
        if mode not in ('spatial', 'mean_channel') or dimension < 1:
            raise ValueError('invalid readout mode/dimension')
        if not 0 < residual_cap <= 1:
            raise ValueError('invalid residual cap')
        self.mode = mode
        self.dimension = dimension
        self.residual_cap = float(residual_cap)
        self.query = nn.Parameter(torch.randn(dimension) * .02)
        self.gain = nn.Parameter(torch.zeros(()))

    def forward(self, global_features, patch_features):
        if global_features.ndim != 2 or patch_features.ndim != 3:
            raise ValueError('expected global BxD and patch BxTxD')
        if patch_features.shape[0] != global_features.shape[0] or patch_features.shape[2] != self.dimension:
            raise ValueError('token batch/dimension mismatch')
        if global_features.shape[1] != self.dimension or patch_features.shape[1] < 1:
            raise ValueError('empty tokens or global dimension mismatch')
        if global_features.device != patch_features.device:
            raise ValueError('global and token device mismatch')
        if self.mode == 'spatial':
            scores = (patch_features.float() @ self.query.float()) / math.sqrt(self.dimension)
            attention = torch.softmax(scores, dim=1)
            residual = (patch_features.float() * attention.unsqueeze(-1)).sum(dim=1)
        else:
            # Equal parameter-count non-spatial control, not old concatenation head.
            residual = patch_features.float().mean(dim=1) * self.query.float()
        scale = self.residual_cap * self.gain.float().tanh()
        return global_features + (scale * residual).to(global_features.dtype)
