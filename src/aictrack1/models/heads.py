"""Small classifier heads for frozen CLIP image features."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


class CosineClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        *,
        initial_scale: float = 10.0,
        learnable_scale: bool = True,
    ) -> None:
        super().__init__()
        if initial_scale <= 0:
            raise ValueError("initial_scale must be positive")
        self.weight = nn.Parameter(torch.empty(num_classes, feature_dim))
        nn.init.normal_(self.weight, std=0.01)
        value = torch.tensor(math.log(initial_scale), dtype=torch.float32)
        if learnable_scale:
            self.log_scale = nn.Parameter(value)
        else:
            self.register_buffer("log_scale", value)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized_features = F.normalize(features, dim=-1)
        normalized_weight = F.normalize(self.weight, dim=-1)
        scale = self.log_scale.exp().clamp(max=100.0)
        return scale * normalized_features @ normalized_weight.t()


def build_classifier_head(
    kind: str,
    feature_dim: int,
    num_classes: int,
    *,
    cosine_initial_scale: float = 10.0,
    cosine_learnable_scale: bool = True,
) -> nn.Module:
    if kind == "linear":
        return LinearClassifier(feature_dim, num_classes)
    if kind == "cosine":
        return CosineClassifier(
            feature_dim,
            num_classes,
            initial_scale=cosine_initial_scale,
            learnable_scale=cosine_learnable_scale,
        )
    raise ValueError(f"unsupported classifier head {kind!r}")
