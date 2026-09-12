from __future__ import annotations

import pytest
import torch

from aictrack1.methods.confidence import visual_prototype_soft_weights


def test_visual_prototype_weights_downweight_only_low_trust_tail() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.0, 1.0],  # suspicious for observed class 0
            [0.0, 1.0],
            [0.1, 0.9],
            [0.2, 0.8],
            [1.0, 0.0],  # suspicious for observed class 1
        ]
    )
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    weights, diagnostics = visual_prototype_soft_weights(
        features,
        labels,
        low_trust_fraction=0.5,
        min_weight=0.4,
    )

    assert weights.shape == labels.shape
    assert torch.isfinite(weights).all()
    assert float(weights.min()) == pytest.approx(0.4)
    assert weights[3] < weights[0]
    assert weights[7] < weights[4]
    assert diagnostics["class_count"] == 2
    assert diagnostics["downweighted_count"] > 0


def test_visual_prototype_weights_require_two_samples_per_class() -> None:
    with pytest.raises(ValueError, match="at least two"):
        visual_prototype_soft_weights(
            torch.eye(3),
            torch.tensor([0, 0, 1]),
        )
