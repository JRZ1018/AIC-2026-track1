"""Training-only, class-aware confidence utilities for noisy-label experiments."""

from __future__ import annotations

import torch


def classwise_rank_noise_scores(losses: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
    """Map losses to tie-aware within-class percentile scores; higher means noisier."""
    if losses.ndim != 1 or class_ids.ndim != 1 or losses.shape != class_ids.shape:
        raise ValueError("losses and class_ids must be matching 1D tensors")
    if not torch.isfinite(losses).all():
        raise ValueError("losses must be finite")
    scores = torch.zeros_like(losses, dtype=torch.float32)
    for class_id in torch.unique(class_ids, sorted=True):
        indices = torch.nonzero(class_ids == class_id, as_tuple=False).flatten()
        values = losses.index_select(0, indices).float()
        if indices.numel() == 1:
            scores[indices] = 0.0
            continue
        order = torch.argsort(values, stable=True)
        sorted_values = values.index_select(0, order)
        ranks = torch.empty_like(sorted_values)
        start = 0
        while start < sorted_values.numel():
            end = start + 1
            while end < sorted_values.numel() and sorted_values[end] == sorted_values[start]:
                end += 1
            ranks[start:end] = (start + end - 1) / 2.0
            start = end
        percentiles = ranks / (indices.numel() - 1)
        scores[indices.index_select(0, order)] = percentiles
    return scores


def select_suspected_noisy(
    noise_scores: torch.Tensor,
    class_ids: torch.Tensor,
    *,
    fraction: float,
    min_keep_per_class: int = 1,
) -> torch.Tensor:
    """Select the highest scores within each observed class while protecting tails."""
    if noise_scores.ndim != 1 or class_ids.ndim != 1 or noise_scores.shape != class_ids.shape:
        raise ValueError("noise_scores and class_ids must be matching 1D tensors")
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0, 1]")
    if min_keep_per_class < 1:
        raise ValueError("min_keep_per_class must be at least 1")
    selected = torch.zeros_like(class_ids, dtype=torch.bool)
    for class_id in torch.unique(class_ids, sorted=True):
        indices = torch.nonzero(class_ids == class_id, as_tuple=False).flatten()
        remove_count = min(round(indices.numel() * fraction), max(0, indices.numel() - min_keep_per_class))
        if remove_count:
            order = torch.argsort(noise_scores.index_select(0, indices), descending=True, stable=True)
            selected[indices.index_select(0, order[:remove_count])] = True
    return selected
