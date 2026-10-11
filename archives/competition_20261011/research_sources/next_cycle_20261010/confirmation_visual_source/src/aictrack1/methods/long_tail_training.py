"""Training-only long-tail controls derived from the current-stage train split."""

from __future__ import annotations

import torch


def train_loss_class_weights(
    class_ids: torch.Tensor,
    *,
    num_classes: int,
    mode: str,
    power: float = 0.5,
    beta: float = 0.99,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    """Build mean-effective-one class weights from the current train split.

    The normalization keeps the average per-sample loss scale at one so this
    experiment changes class contribution rather than silently changing the
    optimizer's effective learning rate.
    """

    labels = class_ids.detach().long().cpu()
    if labels.ndim != 1 or labels.numel() == 0:
        raise ValueError("class_ids must be a non-empty 1D tensor")
    if num_classes < 1 or int(labels.min()) < 0 or int(labels.max()) >= num_classes:
        raise ValueError("class_ids exceed num_classes")
    counts = torch.bincount(labels, minlength=num_classes)
    if bool((counts == 0).any()):
        raise ValueError("class-weighted loss requires every class in the train split")
    if mode == "inverse_frequency":
        if not 0.0 < power <= 1.0:
            raise ValueError("class-weight power must be in (0, 1]")
        weights = counts.float().pow(-float(power))
    elif mode == "effective_number":
        if not 0.0 < beta < 1.0:
            raise ValueError("effective-number beta must be in (0, 1)")
        weights = (1.0 - float(beta)) / (1.0 - torch.pow(float(beta), counts.float()))
    else:
        raise ValueError("class-weight mode must be inverse_frequency or effective_number")
    sample_scale = weights.index_select(0, labels).mean()
    weights = weights / sample_scale
    diagnostics: dict[str, float | int | str] = {
        "source": "current_stage_train_split_only",
        "mode": mode,
        "class_count": int(num_classes),
        "sample_count": int(labels.numel()),
        "class_support_min": int(counts.min()),
        "class_support_max": int(counts.max()),
        "class_weight_min": float(weights.min()),
        "class_weight_max": float(weights.max()),
        "effective_sample_weight_mean": float(weights.index_select(0, labels).mean()),
    }
    if mode == "inverse_frequency":
        diagnostics["power"] = float(power)
    else:
        diagnostics["beta"] = float(beta)
    return weights, diagnostics


def mild_class_balanced_sample_weights(
    class_ids: torch.Tensor,
    *,
    num_classes: int,
    power: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Return mean-one sample weights proportional to ``class_count ** -power``.

    ``power=0`` is natural sampling and ``power=1`` is full inverse-frequency
    balancing. Stage2 screening uses ``0.5`` as a deliberately mild single-variable
    intervention. Only labels from the explicitly supplied training split are used.
    """

    labels = class_ids.detach().long().cpu()
    if labels.ndim != 1 or labels.numel() == 0:
        raise ValueError("class_ids must be a non-empty 1D tensor")
    if num_classes < 1 or int(labels.min()) < 0 or int(labels.max()) >= num_classes:
        raise ValueError("class_ids exceed num_classes")
    if not 0.0 < power <= 1.0:
        raise ValueError("class-balance power must be in (0, 1]")
    counts = torch.bincount(labels, minlength=num_classes)
    if bool((counts == 0).any()):
        raise ValueError("class-balanced sampling requires every class in the train split")
    class_weights = counts.float().pow(-power)
    weights = class_weights.index_select(0, labels)
    weights = weights / weights.mean()
    diagnostics: dict[str, float | int] = {
        "sample_count": int(labels.numel()),
        "class_count": int(num_classes),
        "power": float(power),
        "class_support_min": int(counts.min()),
        "class_support_max": int(counts.max()),
        "sample_weight_min": float(weights.min()),
        "sample_weight_max": float(weights.max()),
        "sample_weight_mean": float(weights.mean()),
    }
    return weights, diagnostics


def train_prior_logit_adjustment(
    class_counts: torch.Tensor,
    *,
    tau: float,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    """Return ``tau * log(train_prior)`` for training-loss logit adjustment."""

    counts = class_counts.detach().float().cpu()
    if counts.ndim != 1 or counts.numel() == 0:
        raise ValueError("class_counts must be a non-empty 1D tensor")
    if not torch.isfinite(counts).all() or bool((counts <= 0).any()):
        raise ValueError("class_counts must be finite and strictly positive")
    if not 0.0 < tau <= 2.0:
        raise ValueError("logit-adjustment tau must be in (0, 2]")
    prior = counts / counts.sum()
    adjustment = float(tau) * prior.log()
    diagnostics: dict[str, float | int | str] = {
        "source": "current_stage_train_split_only",
        "class_count": int(counts.numel()),
        "tau": float(tau),
        "prior_min": float(prior.min()),
        "prior_max": float(prior.max()),
        "adjustment_min": float(adjustment.min()),
        "adjustment_max": float(adjustment.max()),
    }
    return adjustment, diagnostics
