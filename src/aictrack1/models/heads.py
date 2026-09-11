"""Small classifier heads for frozen CLIP image features."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

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


class FrozenFeatureImageClassifier(nn.Module):
    """Apply a frozen-feature head to normalized CLIP image embeddings."""

    def __init__(self, visual: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.visual = visual
        self.classifier = classifier

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = F.normalize(self.visual(images).float(), dim=-1)
        return self.classifier(features)


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


def load_classifier_head_checkpoint(
    path: str | Path,
    *,
    feature_dim: int,
    num_classes: int,
    expected_clip_weight_sha256: str,
    expected_class_mapping_sha256: str,
    expected_split_sha256: str | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a frozen-feature head only after its provenance matches the run."""

    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("classifier head checkpoint must contain a mapping")
    head_config = checkpoint.get("head")
    metadata = checkpoint.get("feature_cache_metadata")
    state = checkpoint.get("model")
    if not isinstance(head_config, dict) or not isinstance(metadata, dict):
        raise ValueError("classifier head checkpoint is missing head or feature metadata")
    if not isinstance(state, dict):
        raise ValueError("classifier head checkpoint is missing model state")
    exact_metadata = {
        "clip_weight_sha256": expected_clip_weight_sha256,
        "class_mapping_sha256": expected_class_mapping_sha256,
    }
    if expected_split_sha256 is not None:
        exact_metadata["split_sha256"] = expected_split_sha256
    for field, expected in exact_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"classifier head checkpoint {field} mismatch: "
                f"{metadata.get(field)!r} != {expected!r}"
            )
    head = build_classifier_head(
        str(head_config.get("kind")),
        feature_dim,
        num_classes,
        cosine_initial_scale=float(head_config.get("initial_scale", 10.0)),
        cosine_learnable_scale=bool(head_config.get("learnable_scale", True)),
    )
    try:
        head.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError("classifier head checkpoint state is incompatible") from error
    return head, checkpoint
