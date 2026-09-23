from __future__ import annotations

import pytest
import torch

from aictrack1.methods.long_tail_training import (
    mild_class_balanced_sample_weights,
    train_loss_class_weights,
    train_prior_logit_adjustment,
)


def test_mild_class_balancing_increases_tail_sampling_without_full_inversion() -> None:
    labels = torch.tensor([0] * 16 + [1] * 4 + [2])
    weights, diagnostics = mild_class_balanced_sample_weights(
        labels,
        num_classes=3,
        power=0.5,
    )
    assert weights.shape == labels.shape
    assert float(weights.mean()) == pytest.approx(1.0)
    assert weights[labels == 2][0] > weights[labels == 1][0] > weights[labels == 0][0]
    assert weights[labels == 2][0] / weights[labels == 0][0] == pytest.approx(4.0)
    assert diagnostics["class_support_min"] == 1
    assert diagnostics["class_support_max"] == 16


def test_class_balancing_rejects_missing_classes_and_invalid_power() -> None:
    with pytest.raises(ValueError, match="every class"):
        mild_class_balanced_sample_weights(torch.tensor([0, 0]), num_classes=2, power=0.5)
    with pytest.raises(ValueError, match="power"):
        mild_class_balanced_sample_weights(torch.tensor([0, 1]), num_classes=2, power=0.0)


def test_train_prior_logit_adjustment_is_train_only_and_favors_rare_margin() -> None:
    adjustment, diagnostics = train_prior_logit_adjustment(
        torch.tensor([100, 25, 4]),
        tau=0.5,
    )
    assert adjustment[0] > adjustment[1] > adjustment[2]
    assert diagnostics["source"] == "current_stage_train_split_only"
    assert diagnostics["tau"] == 0.5
    assert diagnostics["prior_min"] == pytest.approx(4 / 129)


def test_train_prior_logit_adjustment_rejects_empty_class_and_invalid_tau() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        train_prior_logit_adjustment(torch.tensor([2, 0]), tau=0.5)
    with pytest.raises(ValueError, match="tau"):
        train_prior_logit_adjustment(torch.tensor([2, 1]), tau=0.0)


@pytest.mark.parametrize(
    ("mode", "kwargs"),
    [
        ("inverse_frequency", {"power": 0.5}),
        ("effective_number", {"beta": 0.99}),
    ],
)
def test_train_loss_class_weights_favor_tail_and_preserve_loss_scale(mode, kwargs) -> None:
    labels = torch.tensor([0] * 16 + [1] * 4 + [2])
    weights, diagnostics = train_loss_class_weights(
        labels,
        num_classes=3,
        mode=mode,
        **kwargs,
    )
    assert weights[2] > weights[1] > weights[0]
    assert float(weights.index_select(0, labels).mean()) == pytest.approx(1.0)
    assert diagnostics["source"] == "current_stage_train_split_only"
    assert diagnostics["class_support_min"] == 1


def test_train_loss_class_weights_reject_invalid_or_missing_classes() -> None:
    with pytest.raises(ValueError, match="every class"):
        train_loss_class_weights(
            torch.tensor([0, 0]), num_classes=2, mode="inverse_frequency"
        )
    with pytest.raises(ValueError, match="mode"):
        train_loss_class_weights(torch.tensor([0, 1]), num_classes=2, mode="unknown")
