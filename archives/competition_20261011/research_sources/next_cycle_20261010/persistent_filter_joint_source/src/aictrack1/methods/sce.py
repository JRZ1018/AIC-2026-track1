"""Independent hard-label SCE implementation from Wang et al., ICCV 2019.

Source: arXiv:1908.06112v1, equations (5), (7); no upstream code copied.
Author repository audited at ec778df4a15192f8ccee2efb0fce8b94b8e00262.
This implements log(0)=A<0, not label smoothing or a teacher model.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def validate_sce_parameters(alpha: float, beta: float, log_zero: float) -> None:
    if not all(math.isfinite(x) for x in (alpha, beta, log_zero)):
        raise ValueError("SCE parameters must be finite")
    if alpha < 0 or beta < 0 or alpha + beta <= 0:
        raise ValueError("SCE alpha/beta must be nonnegative and not both zero")
    if log_zero >= 0:
        raise ValueError("SCE log_zero must be negative")


def symmetric_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.1,
    beta: float = 1.0,
    log_zero: float = -4.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Return alpha*CE + beta*RCE in FP32 for integer hard labels.

    RCE = -sum_k p_k log(y_k) = -A*(1-p_y), where log(0)=A.
    Using this identity avoids allocating a batch-by-750 one-hot tensor.
    It is deliberately not renormalized, probability-clipped, or detached.
    AC weights and their detached CE history are handled by the trainer.
    """
    validate_sce_parameters(alpha, beta, log_zero)
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("unsupported SCE reduction")
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("SCE expects [batch, classes] logits and [batch] targets")
    if targets.numel() == 0 or logits.shape[1] < 2:
        raise ValueError("SCE requires a non-empty batch and at least two classes")
    if not logits.is_floating_point() or targets.dtype != torch.long:
        raise ValueError("SCE expects floating logits and int64 hard targets")
    if targets.device != logits.device:
        raise ValueError("SCE logits and targets must be on the same device")
    target_log_probability = F.log_softmax(logits.float(), dim=-1).gather(
        1, targets[:, None]
    ).squeeze(1)
    # -expm1(log p_y) is stable when p_y is very close to one.
    reverse_ce = log_zero * torch.expm1(target_log_probability)
    losses = alpha * (-target_log_probability) + beta * reverse_ce
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()
