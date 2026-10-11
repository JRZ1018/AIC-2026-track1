"""Balanced Softmax loss using only the current training partition's counts."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def balanced_softmax_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, class_counts: torch.Tensor
) -> torch.Tensor:
    """Apply the canonical ``log(n_c)`` offset inside training CE only.

    The offset is deliberately absent from evaluation logits.  Accumulate the
    softmax in float32 so small class counts remain stable under CUDA AMP.
    """

    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("expected logits [N, C] and targets [N]")
    if class_counts.ndim != 1 or logits.shape[1] != class_counts.numel():
        raise ValueError("class_counts must contain one count per logit class")
    if not torch.isfinite(class_counts).all() or bool((class_counts <= 0).any()):
        raise ValueError("every training class must have a positive finite count")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    return F.cross_entropy(
        logits.float() + class_counts.to(logits.device, dtype=torch.float32).log(),
        targets,
    )
