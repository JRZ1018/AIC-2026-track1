from __future__ import annotations

import numpy as np
import pytest
import torch

from aictrack1.evaluation.feature_geometry import (
    centroid_validation_scores,
    class_centroids,
    grouped_bootstrap_delta,
    nearest_neighbor_label_agreement,
)


def test_centroid_scores_use_train_only_and_preserve_true_margin() -> None:
    train = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1])
    centroids = class_centroids(train, labels, num_classes=2)
    assert torch.allclose(centroids.norm(dim=1), torch.ones(2))
    validation = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    predictions, true_scores, other_scores, margins = centroid_validation_scores(
        validation, torch.tensor([0, 1, 1]), centroids, device=torch.device("cpu"), batch_size=2
    )
    assert predictions.tolist() == [0, 1, 0]
    assert margins[0] > 0 and margins[1] > 0 and margins[2] < 0
    assert torch.allclose(margins, true_scores - other_scores)


def test_centroid_rejects_unsupported_classes() -> None:
    with pytest.raises(ValueError, match="every class"):
        class_centroids(torch.eye(2), torch.tensor([0, 0]), num_classes=2)


def test_nearest_neighbor_agreement_is_chunk_invariant() -> None:
    train = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [-1.0, 0.0]])
    labels = torch.tensor([0, 0, 1, 1, 2])
    validation = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    expected = torch.tensor([0, 1, 2])
    first = nearest_neighbor_label_agreement(
        train, labels, validation, expected, device=torch.device("cpu"), k=3, batch_size=1
    )
    second = nearest_neighbor_label_agreement(
        train, labels, validation, expected, device=torch.device("cpu"), k=3, batch_size=3
    )
    assert torch.equal(first[0], torch.ones(3, dtype=torch.bool))
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


def test_group_bootstrap_is_deterministic_and_requires_aligned_rows() -> None:
    baseline = np.array([0.0, 1.0, 0.0, 1.0])
    candidate = np.array([1.0, 1.0, 0.0, 1.0])
    groups = ["a", "a", "b", "c"]
    one = grouped_bootstrap_delta(baseline, candidate, groups, seed=13, replicates=100)
    two = grouped_bootstrap_delta(baseline, candidate, groups, seed=13, replicates=100)
    assert one == two
    assert one["delta"] == 0.25
    assert one["group_count"] == 3
    with pytest.raises(ValueError, match="incompatible shapes"):
        grouped_bootstrap_delta(baseline, candidate[:2], groups, seed=13)
