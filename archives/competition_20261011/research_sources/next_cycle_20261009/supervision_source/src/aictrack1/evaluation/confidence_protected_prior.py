"""Pre-registered P20 protection for P7's single-checkpoint balanced prior."""

from __future__ import annotations

import math

import torch


def confidence_protected_predictions(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    median_margin: float,
    bias_strength: float = 0.75,
    margin_multiple: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return P7 predictions, P20 predictions and the frozen protection mask.

    The caller must fit ``bias`` with P7's existing uniform-soft algorithm.  The
    margin scale is the same unlabeled median already returned by that fit.
    """

    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("logits must have shape [samples, classes>=2]")
    if bias.shape != (logits.shape[1],):
        raise ValueError("bias must have one value per class")
    if not torch.isfinite(logits).all() or not torch.isfinite(bias).all():
        raise ValueError("logits and bias must be finite")
    if not math.isfinite(median_margin) or median_margin <= 0:
        raise ValueError("median_margin must be finite and positive")
    if not math.isfinite(bias_strength) or bias_strength != 0.75:
        raise ValueError("P20 freezes P7 bias strength at 0.75")
    if not math.isfinite(margin_multiple) or margin_multiple != 2.0:
        raise ValueError("P20 freezes the protection multiple at 2.0")

    top2 = logits.topk(2, dim=1).values
    protected = (top2[:, 0] - top2[:, 1]) >= margin_multiple * median_margin
    p7 = (logits + bias_strength * bias).argmax(dim=1)
    original = logits.argmax(dim=1)
    p20 = torch.where(protected, original, p7)
    return p7, p20, protected
