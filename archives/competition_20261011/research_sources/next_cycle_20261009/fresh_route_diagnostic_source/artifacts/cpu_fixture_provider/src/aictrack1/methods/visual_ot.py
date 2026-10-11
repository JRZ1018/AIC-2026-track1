"""Train-only empirical-marginal visual OT confidence from frozen CLIP features."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def empirical_visual_ot_weights(
    features: torch.Tensor,
    class_ids: torch.Tensor,
    *,
    device: torch.device,
    temperature: float = 0.05,
    iterations: int = 30,
    low_trust_fraction: float = 0.10,
    min_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    """Apply empirical long-tail column marginals; downweight global bottom posterior.

    All inputs must come from the current official train partition. The observed
    class's prototype uses leave-one-out features, avoiding self-similarity bias.
    The output contains one finite, positive static weight per input row.
    """

    if features.ndim != 2 or class_ids.ndim != 1 or len(features) != len(class_ids) or not len(features):
        raise ValueError("features and class_ids must be nonempty matching 2D/1D tensors")
    if not torch.isfinite(features).all():
        raise ValueError("features contain non-finite values")
    if not 0 < temperature <= 1 or iterations < 1 or not 0 < low_trust_fraction < 1 or not 0 < min_weight < 1:
        raise ValueError("invalid visual OT hyperparameters")
    labels = class_ids.detach().long().cpu()
    if int(labels.min()) < 0:
        raise ValueError("class_ids must be nonnegative")
    class_count = int(labels.max()) + 1
    counts = torch.bincount(labels, minlength=class_count)
    if bool((counts < 2).any()):
        raise ValueError("leave-one-out OT requires at least two train rows per class")
    x = F.normalize(features.detach().float().to(device), dim=1)
    y = labels.to(device)
    counts_device = counts.float().to(device)
    sums = torch.zeros((class_count, x.shape[1]), dtype=torch.float32, device=device)
    sums.index_add_(0, y, x)
    prototypes = F.normalize(sums / counts_device[:, None], dim=1)
    similarities = x @ prototypes.T
    rows = torch.arange(len(labels), device=device)
    similarities[rows, y] = (x * F.normalize(sums[y] - x, dim=1)).sum(1)
    scaled = similarities / float(temperature)
    kernel = torch.exp(scaled - scaled.max(1, keepdim=True).values)
    desired = counts_device / len(labels)
    scaling = torch.ones(class_count, dtype=torch.float32, device=device)
    for _ in range(iterations):
        denominator = (kernel @ scaling).clamp_min(1e-20)
        empirical = ((kernel * scaling) / denominator[:, None]).mean(0).clamp_min(1e-20)
        scaling = (scaling * desired / empirical).clamp(1e-12, 1e12)
    denominator = (kernel @ scaling).clamp_min(1e-20)
    posterior = (kernel[rows, y] * scaling[y] / denominator).cpu()
    empirical = ((kernel * scaling) / denominator[:, None]).mean(0)
    marginal_l1 = float((empirical - desired).abs().sum().item())
    if not torch.isfinite(posterior).all() or not math.isfinite(marginal_l1) or marginal_l1 > 0.05:
        raise RuntimeError(f"visual OT did not converge: marginal L1={marginal_l1}")
    selected_count = max(1, round(len(labels) * low_trust_fraction))
    selected = torch.argsort(posterior, stable=True)[:selected_count]
    weights = torch.ones(len(labels), dtype=torch.float32)
    weights[selected] = float(min_weight)
    return weights, {
        "source": "current_stage_train_partition_only",
        "sample_count": len(labels),
        "class_count": class_count,
        "class_support_min": int(counts.min()),
        "class_support_max": int(counts.max()),
        "temperature": float(temperature),
        "iterations": iterations,
        "low_trust_fraction": float(low_trust_fraction),
        "min_weight": float(min_weight),
        "downweighted_count": selected_count,
        "weight_mean": float(weights.mean()),
        "posterior_mean": float(posterior.mean()),
        "sinkhorn_marginal_l1": marginal_l1,
    }
