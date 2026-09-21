from __future__ import annotations

import pytest
import torch

from aictrack1.methods.confidence import (
    ClasswiseHistoryWeighting,
    classwise_rank_noise_scores,
    select_suspected_noisy,
    weighted_cross_entropy,
)


def test_classwise_scores_and_tail_protection() -> None:
    losses = torch.tensor([0.1, 0.2, 2.0, 0.3, 0.4, 3.0, 5.0])
    classes = torch.tensor([0, 0, 0, 1, 1, 1, 2])
    scores = classwise_rank_noise_scores(losses, classes)
    selected = select_suspected_noisy(scores, classes, fraction=1 / 3, min_keep_per_class=2)
    assert selected.tolist() == [False, False, True, False, False, True, False]
    assert not selected[classes == 2].any()


def test_tied_losses_receive_equal_scores() -> None:
    scores = classwise_rank_noise_scores(torch.ones(4), torch.tensor([0, 0, 0, 0]))
    assert torch.equal(scores, torch.full((4,), 0.5))


def test_history_weighting_is_classwise_soft_and_order_independent() -> None:
    state = ClasswiseHistoryWeighting(
        torch.tensor([10, 30, 20, 40, 50, 60]),
        torch.tensor([0, 0, 0, 1, 1, 1]),
        momentum=0.5,
        history_mix=0.5,
        min_weight=0.2,
    )
    diagnostics = state.update_epoch(
        torch.tensor([60, 20, 40, 10, 50, 30]),
        torch.tensor([3.0, 2.0, 0.1, 0.1, 1.0, 1.0]),
        torch.tensor([0.05, 0.10, 0.95, 0.95, 0.50, 0.50]),
        torch.tensor([0, 1, 1, 0, 1, 0]),
    )
    weights = state.weights_for(torch.tensor([10, 30, 20, 40, 50, 60]))
    assert torch.all((weights >= 0.2) & (weights <= 1.0))
    assert weights[0] > weights[1] > weights[2]
    assert weights[3] > weights[4] > weights[5]
    assert diagnostics["epochs_updated"] == 1
    assert diagnostics["class_mean_min"] == diagnostics["class_mean_max"]


def test_history_weighting_rewards_stable_observed_label_agreement() -> None:
    state = ClasswiseHistoryWeighting(
        torch.arange(4),
        torch.zeros(4, dtype=torch.long),
        momentum=0.5,
        history_mix=1.0,
        min_weight=0.1,
    )
    losses = torch.ones(4)
    state.update_epoch(
        torch.arange(4), losses, torch.tensor([0.9, 0.7, 0.4, 0.2]), torch.tensor([0, 0, 1, 2])
    )
    state.update_epoch(
        torch.arange(4), losses, torch.tensor([0.9, 0.6, 0.3, 0.2]), torch.tensor([0, 1, 2, 2])
    )
    weights = state.weights_for(torch.arange(4))
    assert weights[0] > weights[1] > weights[2] > weights[3]


def test_history_weighting_checkpoint_round_trip_and_strict_coverage() -> None:
    indices = torch.tensor([2, 5, 8])
    classes = torch.tensor([0, 0, 1])
    state = ClasswiseHistoryWeighting(indices, classes)
    with pytest.raises(ValueError, match="every training sample"):
        state.update_epoch(indices[:2], torch.ones(2), torch.ones(2), torch.zeros(2, dtype=torch.long))
    state.update_epoch(indices, torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.9, 0.8, 0.7]), classes)
    restored = ClasswiseHistoryWeighting(indices, classes)
    restored.load_state_dict(state.state_dict())
    assert torch.equal(restored.weights, state.weights)
    assert restored.diagnostics() == state.diagnostics()


def test_history_weighting_reports_classwise_effective_quality() -> None:
    indices = torch.tensor([10, 20, 30, 40, 50, 60])
    classes = torch.tensor([0, 0, 0, 1, 1, 1])
    losses = torch.tensor([0.1, 0.2, 0.3, 10.0, 20.0, 30.0])
    state = ClasswiseHistoryWeighting(
        indices,
        classes,
        momentum=0.0,
        history_mix=0.0,
        min_weight=0.02,
        magnitude_mix=1.0,
    )
    state.update_epoch(indices, losses, torch.full((6,), 0.5), classes)

    profile = state.effective_quality_profile(indices, losses)

    assert profile["class_count"] == 2
    assert profile["observation_count"] == 6
    assert profile["classes"][0]["raw_support"] == 3
    assert profile["classes"][1]["raw_support"] == 3
    assert profile["classes"][0]["mean_weight"] > profile["classes"][1]["mean_weight"]
    assert profile["classes"][1]["effective_quality"] < 1.0


def test_effective_quality_profile_requires_complete_unique_epoch() -> None:
    indices = torch.tensor([2, 5, 8])
    state = ClasswiseHistoryWeighting(indices, torch.tensor([0, 0, 1]))
    with pytest.raises(ValueError, match="every training sample"):
        state.effective_quality_profile(indices[:2], torch.ones(2))
    with pytest.raises(ValueError, match="duplicate samples"):
        state.effective_quality_profile(
            torch.tensor([2, 2, 8]), torch.ones(3)
        )


def test_magnitude_mix_keeps_between_class_loss_magnitude() -> None:
    """The classwise percentile alone cannot see that one class is noisier than another.

    Screening 2 measured this on the audited conflict ground truth: the classwise
    percentile separates conflicts with AUC 0.642, the global loss ranking with 0.790.
    """
    indices = torch.arange(6)
    # Two classes with clearly different loss scales, each internally ordered.
    classes = torch.tensor([0, 0, 0, 1, 1, 1])
    losses = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    probabilities = torch.full((6,), 0.5)
    predictions = torch.zeros(6, dtype=torch.long)

    classwise = ClasswiseHistoryWeighting(indices, classes, momentum=0.0, history_mix=0.5, min_weight=0.1)
    classwise.update_epoch(indices, losses, probabilities, predictions)
    baseline = classwise.weights_for(indices)

    mixed = ClasswiseHistoryWeighting(
        indices, classes, momentum=0.0, history_mix=0.5, min_weight=0.1, magnitude_mix=1.0
    )
    mixed.update_epoch(indices, losses, probabilities, predictions)
    blended = mixed.weights_for(indices)

    # Pure classwise: identical within-class ordering in both classes, so the class with
    # the larger absolute losses is treated exactly like the easy class.
    assert torch.allclose(baseline[:3], baseline[3:])
    assert abs(float(baseline[:3].mean() - baseline[3:].mean())) < 1e-6
    # Pure global ranking: ranks come from the whole split, so the easy class is treated
    # as clean and the hard class as noisy, even though both are ordered identically
    # inside themselves. The noisiest easy sample beats the cleanest hard sample.
    assert blended[2] > blended[3]
    assert torch.all(blended[:3] > baseline[:3])
    assert torch.all(blended[3:] < baseline[3:])


def test_magnitude_mix_default_reproduces_the_classwise_only_weights() -> None:
    indices = torch.arange(6)
    classes = torch.tensor([0, 0, 0, 1, 1, 1])
    losses = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    probabilities = torch.full((6,), 0.5)
    predictions = torch.zeros(6, dtype=torch.long)

    default = ClasswiseHistoryWeighting(indices, classes, momentum=0.0, history_mix=0.5, min_weight=0.1)
    explicit = ClasswiseHistoryWeighting(
        indices, classes, momentum=0.0, history_mix=0.5, min_weight=0.1, magnitude_mix=0.0
    )
    default.update_epoch(indices, losses, probabilities, predictions)
    explicit.update_epoch(indices, losses, probabilities, predictions)
    assert torch.equal(default.weights, explicit.weights)
    assert default.diagnostics()["magnitude_mix"] == 0.0


def test_magnitude_mix_is_validated_and_round_trips_through_the_checkpoint() -> None:
    indices = torch.arange(4)
    classes = torch.tensor([0, 0, 1, 1])
    with pytest.raises(ValueError, match="magnitude_mix"):
        ClasswiseHistoryWeighting(indices, classes, magnitude_mix=1.5)
    with pytest.raises(ValueError, match="magnitude_mix"):
        ClasswiseHistoryWeighting(indices, classes, magnitude_mix=-0.1)

    state = ClasswiseHistoryWeighting(indices, classes, magnitude_mix=0.5, min_weight=0.1)
    state.update_epoch(indices, torch.tensor([0.1, 0.2, 0.3, 0.4]), torch.ones(4), classes)
    restored = ClasswiseHistoryWeighting(indices, classes, magnitude_mix=0.5, min_weight=0.1)
    restored.load_state_dict(state.state_dict())
    assert torch.equal(restored.weights, state.weights)

    mismatched = ClasswiseHistoryWeighting(indices, classes, magnitude_mix=0.25, min_weight=0.1)
    with pytest.raises(ValueError, match="magnitude_mix"):
        mismatched.load_state_dict(state.state_dict())


def test_weighted_cross_entropy_is_normalized_and_validated() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    targets = torch.tensor([0, 1])
    weights = torch.tensor([0.25, 1.0])
    per_sample = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    expected = (per_sample * weights).sum() / weights.sum()
    assert torch.allclose(weighted_cross_entropy(logits, targets, weights), expected)
    with pytest.raises(ValueError, match="strictly positive"):
        weighted_cross_entropy(logits, targets, torch.tensor([0.0, 1.0]))
