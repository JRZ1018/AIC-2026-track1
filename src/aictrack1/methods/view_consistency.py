"""Independent same-model JS formula, arXiv:1912.02781v2 Eq. (2).

Concept source: google-research/augmix 9b9824c7c19bf7e72df2d085d97b99b3bfb00ba4
(Apache-2.0). No upstream code copied; not an AugMix pipeline or teacher.
"""
from __future__ import annotations

import math
import torch


def view_jensen_shannon(a: torch.Tensor, b: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
    """Symmetric two-view divergence; both branches retain gradients."""
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] == 0 or a.shape[1] < 2:
        raise ValueError("Two equal nonempty [batch, classes>=2] tensors required")
    if not a.is_floating_point() or not b.is_floating_point() or a.device != b.device:
        raise ValueError("Floating logits on the same device required")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("Unsupported reduction")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Finite logits required")
    la = a.float().log_softmax(-1)
    lb = b.float().log_softmax(-1)
    lm = torch.logaddexp(la, lb) - math.log(2.0)
    values = .5 * ((la.exp() * (la - lm)).sum(-1) + (lb.exp() * (lb - lm)).sum(-1))
    if reduction == "none":
        return values
    return values.mean() if reduction == "mean" else values.sum()
