"""VC1 train-only same-step consistency, without a teacher or temporal state.

Independent implementation; conceptual source is AugMix (1912.02781v2,
google-research/augmix 9b9824c7c19bf7e72df2d085d97b99b3bfb00ba4, Apache-2.0).
This is a two-view current-model objective, not the full AugMix algorithm.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from aictrack1.data.indexed_images import IndexedImageDataset
from aictrack1.methods.view_consistency import view_jensen_shannon


PROTOCOL = {
    "protocol": "vc1-current-center-strong-js-v1",
    "coefficient": 1.0,
    "temperature": 1.0,
    "second_view": "current-center336",
    "both_branches_gradient": True,
    "confidence_observation": "original-strong-only",
    "teacher": False,
    "temporal_updates": 0,
}


def validate_protocol(setting: dict[str, Any] | None) -> bool:
    if setting is None:
        return False
    if (not isinstance(setting, dict) or setting != PROTOCOL
            or any(type(setting[key]) is not type(value) for key, value in PROTOCOL.items())):
        raise ValueError("VC1 requires the pre-registered exact protocol")
    return True


class CenterPairedTrainingDataset(Dataset):
    """Keep the original seeded augmentation unchanged and add a center tensor.

    Only fold-train rows are accepted. The secondary decode does not consume
    the original transform's random stream. No image/token cache is kept.
    """

    def __init__(self, primary: IndexedImageDataset, center_transform: Any) -> None:
        if not primary.records or any(
            row.source != "train" or row.partition != "train" or row.internal_class_index is None
            for row in primary.records
        ):
            raise ValueError("VC1 accepts only labeled fold-train records")
        if not getattr(primary.transform, "stochastic", False):
            raise ValueError("Original strong train transform required")
        if getattr(center_transform, "stochastic", True) or getattr(center_transform, "split", None) != "validation":
            raise ValueError("Deterministic center transform required, on train rows only")
        self.primary = primary
        self.records = primary.records
        self.center = IndexedImageDataset(
            primary.records, center_transform,
            allow_truncated_images=primary.allow_truncated_images,
        )

    def set_epoch(self, epoch: int) -> None:
        self.primary.set_epoch(epoch)
        self.center.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.primary)

    def __getitem__(self, index: int) -> dict[str, Any]:
        first, second = self.primary[index], self.center[index]
        for field in ("label", "sample_index", "sample_id", "relative_path", "partition"):
            if first[field] != second[field]:
                raise ValueError("Paired train record alignment drift")
        return {**first, "consistency_center_image": second["image"]}


def attach_center_view(dataset: IndexedImageDataset, center_transform: Any,
                       setting: dict[str, Any] | None) -> Dataset:
    if not validate_protocol(setting):
        return dataset
    return CenterPairedTrainingDataset(dataset, center_transform)


def add_current_view_loss(supervised_loss: torch.Tensor, strong_logits: torch.Tensor,
                          center_logits: torch.Tensor | None,
                          setting: dict[str, Any] | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Add unweighted JS; original SCE/AC supervision and observer stay separate."""
    if not validate_protocol(setting):
        if center_logits is not None:
            raise ValueError("Disabled VC1 must not evaluate a secondary branch")
        return supervised_loss, None
    if supervised_loss.ndim != 0 or center_logits is None:
        raise ValueError("Scalar supervised loss and current center logits required")
    divergence = view_jensen_shannon(strong_logits, center_logits)
    return supervised_loss + divergence, divergence


def protocol_state(setting: dict[str, Any] | None) -> dict[str, Any] | None:
    return dict(PROTOCOL) if validate_protocol(setting) else None


def verify_protocol_state(state: dict[str, Any] | None, setting: dict[str, Any] | None) -> None:
    if state is not None:
        validate_protocol(state)
    if state != protocol_state(setting):
        raise ValueError("VC1 checkpoint protocol drift")
