"""Frozen class-balanced inference prior shared by validation and final inference."""

from __future__ import annotations

import math

import torch


def fit_uniform_soft_bias(
    logits: torch.Tensor,
    *,
    temperature_multiplier: float,
    iterations: int = 60,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Fit one additive class bias whose soft prediction mass is uniform.

    The temperature is expressed relative to the median top1/top2 margin so the
    frozen setting transfers across folds without looking at labels or tuning on
    test.  The returned bias is in the original logit units.
    """

    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < 2:
        raise ValueError("logits must be a nonempty [samples, classes] matrix")
    if temperature_multiplier <= 0 or iterations < 1:
        raise ValueError("temperature multiplier and iterations must be positive")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    top2 = torch.topk(logits, k=2, dim=1).values
    median_margin = float((top2[:, 0] - top2[:, 1]).median().item())
    scale = max(median_margin * temperature_multiplier, 1e-4)
    normalized = logits / scale
    bias = torch.zeros(logits.shape[1], dtype=logits.dtype, device=logits.device)
    target_log_mass = -math.log(logits.shape[1])
    max_log_mass_error = float("inf")
    completed = 0
    for completed in range(1, iterations + 1):
        mass = torch.softmax(normalized + bias, dim=1).mean(dim=0).clamp_min(1e-12)
        correction = target_log_mass - mass.log()
        bias = bias + correction
        bias = bias - bias.mean()
        max_log_mass_error = float(correction.abs().max().item())
        if max_log_mass_error <= 1e-4:
            break
    return bias * scale, {
        "median_top1_top2_margin": median_margin,
        "temperature_scale": scale,
        "iterations": completed,
        "max_log_mass_error": max_log_mass_error,
    }
