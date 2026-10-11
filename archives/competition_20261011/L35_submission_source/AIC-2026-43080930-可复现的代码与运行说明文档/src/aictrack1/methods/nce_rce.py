"""Independent hard-label NCE+RCE loss based on Ma et al., ICML 2020.

Reference: https://proceedings.mlr.press/v119/ma20c.html, Eqs. (1), (2), (4), (6).
This is a minimal formula implementation; no upstream training code is copied.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def validate_nce_rce_parameters(alpha: float, beta: float, log_zero: float) -> None:
    if not all(math.isfinite(value) for value in (alpha, beta, log_zero)):
        raise ValueError("NCE+RCE parameters must be finite")
    if alpha <= 0 or beta <= 0:
        raise ValueError("NCE+RCE alpha and beta must be positive")
    if log_zero >= 0:
        raise ValueError("NCE+RCE log_zero must be negative")


def normalized_cross_entropy_plus_reverse_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 750.0,
    beta: float = 0.1,
    log_zero: float = -4.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Return alpha*NCE + beta*RCE for integer hard labels in FP32.

    NCE is CE divided by the sum of CE over all possible one-hot labels for the
    same example. RCE uses log(0)=log_zero and its equivalent -log_zero*(1-p_y)
    expression to avoid a dense one-hot tensor. Alpha is an explicit config
    parameter; this Stage2 screen scales the paper's 50-class WebVision alpha=50
    by 750/50 to account for NCE's class-sum denominator. The scaling is frozen,
    not tuned on validation or test data.
    """
    validate_nce_rce_parameters(alpha, beta, log_zero)
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("unsupported NCE+RCE reduction")
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("NCE+RCE expects [batch, classes] logits and [batch] targets")
    if targets.numel() == 0 or logits.shape[1] < 2:
        raise ValueError("NCE+RCE requires a non-empty batch and at least two classes")
    if not logits.is_floating_point() or targets.dtype != torch.long:
        raise ValueError("NCE+RCE expects floating logits and int64 hard targets")
    if targets.device != logits.device:
        raise ValueError("NCE+RCE logits and targets must be on the same device")
    if bool((targets < 0).any()) or bool((targets >= logits.shape[1]).any()):
        raise ValueError("NCE+RCE target outside class range")
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    target_log_probability = log_probabilities.gather(1, targets[:, None]).squeeze(1)
    numerator = -target_log_probability
    denominator = -log_probabilities.sum(dim=-1)
    normalized_ce = numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny)
    reverse_ce = -log_zero * (-torch.expm1(target_log_probability))
    losses = alpha * normalized_ce + beta * reverse_ce
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()
