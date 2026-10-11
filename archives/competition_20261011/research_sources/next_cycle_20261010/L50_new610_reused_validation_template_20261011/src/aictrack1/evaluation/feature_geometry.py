"""Train-only feature geometry diagnostics for one fixed validation fold."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def class_centroids(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    num_classes: int,
) -> torch.Tensor:
    """Return normalized observed-label centroids without validation fitting."""

    if train_features.ndim != 2 or train_labels.shape != (len(train_features),):
        raise ValueError("train features and labels have incompatible shapes")
    if len(train_features) == 0 or (train_labels < 0).any() or (train_labels >= num_classes).any():
        raise ValueError("train labels must cover valid nonempty class indices")
    sums = torch.zeros(num_classes, train_features.shape[1], dtype=torch.float32)
    sums.index_add_(0, train_labels.long(), train_features.float())
    counts = torch.bincount(train_labels.long(), minlength=num_classes)
    if (counts == 0).any():
        raise ValueError("every class needs train support for centroid diagnostics")
    centroids = sums / counts[:, None]
    if (centroids.norm(dim=1) <= 0).any():
        raise ValueError("a class centroid has zero norm")
    return F.normalize(centroids, dim=1)


@torch.inference_mode()
def centroid_validation_scores(
    features: torch.Tensor,
    labels: torch.Tensor,
    centroids: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return predicted classes, true similarity, nearest-wrong similarity, margins."""

    if len(features) != len(labels) or features.ndim != 2:
        raise ValueError("validation features and labels have incompatible shapes")
    if not 0 < batch_size or centroids.shape[0] < 2:
        raise ValueError("invalid batch size or class count")
    centroids_device = centroids.to(device)
    all_predictions, all_true, all_other, all_margins = [], [], [], []
    for feature_chunk, label_chunk in zip(
        features.split(batch_size), labels.split(batch_size), strict=True
    ):
        scores = F.normalize(feature_chunk.to(device).float(), dim=1) @ centroids_device.T
        targets = label_chunk.to(device).long()
        true_scores = scores.gather(1, targets[:, None]).squeeze(1)
        predictions = scores.argmax(dim=1)
        scores.scatter_(1, targets[:, None], -torch.inf)
        other_scores = scores.max(dim=1).values
        all_predictions.append(predictions.cpu())
        all_true.append(true_scores.cpu())
        all_other.append(other_scores.cpu())
        all_margins.append((true_scores - other_scores).cpu())
    return tuple(torch.cat(parts) for parts in (all_predictions, all_true, all_other, all_margins))


@torch.inference_mode()
def nearest_neighbor_label_agreement(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    *,
    device: torch.device,
    k: int = 5,
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare validation observed labels with top-1 and top-k train neighbors."""

    if k < 1 or len(train_features) < k or len(train_features) != len(train_labels):
        raise ValueError("invalid nearest-neighbor train population")
    if len(validation_features) != len(validation_labels) or batch_size < 1:
        raise ValueError("invalid nearest-neighbor validation population")
    train_device = F.normalize(train_features.to(device).float(), dim=1)
    train_labels_device = train_labels.to(device).long()
    top1, topk_fraction = [], []
    for features, labels in zip(
        validation_features.split(batch_size), validation_labels.split(batch_size), strict=True
    ):
        similarities = F.normalize(features.to(device).float(), dim=1) @ train_device.T
        positions = similarities.topk(k, dim=1).indices
        matches = train_labels_device[positions].eq(labels.to(device).long()[:, None])
        top1.append(matches[:, 0].cpu())
        topk_fraction.append(matches.float().mean(dim=1).cpu())
    return torch.cat(top1), torch.cat(topk_fraction)


def grouped_bootstrap_delta(
    baseline: np.ndarray,
    candidate: np.ndarray,
    group_ids: list[str],
    *,
    seed: int,
    replicates: int = 500,
) -> dict[str, float | int]:
    """Paired group bootstrap of a mean difference, conditional on trained features."""

    first = np.asarray(baseline, dtype=np.float64)
    second = np.asarray(candidate, dtype=np.float64)
    if first.ndim != 1 or first.shape != second.shape or len(first) != len(group_ids):
        raise ValueError("paired values and group IDs have incompatible shapes")
    if not len(first) or replicates < 2 or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("bootstrap requires finite nonempty paired values and >=2 replicates")
    groups: dict[str, int] = {}
    codes = np.empty(len(group_ids), dtype=np.int32)
    for index, group in enumerate(group_ids):
        if group not in groups:
            groups[group] = len(groups)
        codes[index] = groups[group]
    count = np.bincount(codes, minlength=len(groups))
    sums = np.bincount(codes, weights=second - first, minlength=len(groups))
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, len(groups), size=len(groups))
        draws[index] = sums[sampled].sum() / count[sampled].sum()
    return {
        "delta": float((second - first).mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "group_count": len(groups),
        "replicates": replicates,
    }
