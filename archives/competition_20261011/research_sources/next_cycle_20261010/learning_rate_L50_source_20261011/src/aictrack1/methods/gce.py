"""Generalized cross entropy from Zhang and Sabuncu, NeurIPS 2018."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def generalized_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    q: float = 0.7,
    reduction: str = "mean",
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute (1 - p_y**q) / q with a numerically stable target probability."""

    if not 0 < q <= 1:
        raise ValueError("GCE q must be in (0, 1]")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("unsupported GCE reduction")
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("GCE expects [batch, classes] logits and [batch] targets")
    if targets.numel() == 0:
        raise ValueError("GCE requires a non-empty batch")
    log_probability = F.log_softmax(logits.float(), dim=-1)
    target_probability = log_probability.gather(1, targets[:, None]).squeeze(1).exp()
    losses = (1.0 - target_probability.clamp_min(eps).pow(q)) / q
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()
