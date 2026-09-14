from __future__ import annotations

import pytest
import torch

from aictrack1.evaluation.long_tail import (
    HEAD,
    MID,
    TAIL,
    long_tail_diagnostics,
    per_class_recall,
    retention_diagnostics,
    retention_gate,
    stratify_classes,
)


def test_stratify_classes_is_deterministic_and_contiguous() -> None:
    counts = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    first = stratify_classes(counts, head_fraction=0.2, mid_fraction=0.3, seed=7)
    second = stratify_classes(counts, head_fraction=0.2, mid_fraction=0.3, seed=7)
    assert first.assignments == second.assignments
    assert first.counts == {HEAD: 2, MID: 3, TAIL: 5}
    assert first.assignments[0] == HEAD
    assert first.assignments[-1] == TAIL


def test_stratify_classes_breaks_ties_by_class_index() -> None:
    result = stratify_classes([5, 5, 5, 5], head_fraction=0.25, mid_fraction=0.25, seed=1)
    assert result.assignments == (HEAD, MID, TAIL, TAIL)


def test_stratify_classes_rejects_impossible_fractions() -> None:
    with pytest.raises(ValueError, match="below 1"):
        stratify_classes([1, 2, 3], head_fraction=0.5, mid_fraction=0.5, seed=1)
    with pytest.raises(ValueError, match="at least three"):
        stratify_classes([1, 2], head_fraction=0.2, mid_fraction=0.3, seed=1)


def test_per_class_recall_marks_absent_classes() -> None:
    recall, support = per_class_recall(
        torch.tensor([0, 1]), torch.tensor([0, 0]), num_classes=3
    )
    assert recall[0] == pytest.approx(0.5)
    assert recall[1] is None
    assert recall[2] is None
    assert support == {0: 2, 1: 0, 2: 0}
    restricted, restricted_support = per_class_recall(
        torch.tensor([0, 1]), torch.tensor([0, 0]), num_classes=3, present_classes=[0, 1]
    )
    assert set(restricted) == {0, 1}
    assert restricted_support == {0: 2, 1: 0}


def test_per_class_recall_rejects_out_of_range_classes() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        per_class_recall(torch.tensor([3]), torch.tensor([0]), num_classes=3)
    with pytest.raises(ValueError, match="non-negative"):
        per_class_recall(torch.tensor([-1]), torch.tensor([0]), num_classes=3)
    with pytest.raises(ValueError, match="equal shape"):
        per_class_recall(torch.tensor([0, 1]), torch.tensor([0]), num_classes=3)


def test_long_tail_diagnostics_reports_strata_and_worst_class() -> None:
    stratification = stratify_classes(
        [10, 8, 6, 4, 2, 1], head_fraction=0.5, mid_fraction=0.1667, seed=3
    )
    # Two samples per class; only tail classes are predicted incorrectly.
    targets = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    predictions = torch.tensor([0, 0, 1, 1, 2, 2, 3, 0, 0, 0, 0, 0])
    result = long_tail_diagnostics(
        predictions,
        targets,
        num_classes=6,
        stratification=stratification,
        min_stratum_samples=2,
    )
    assert result["sample_count"] == 12
    assert result["evaluated_class_count"] == 6
    assert result["unstable_strata"] == []
    assert result["strata"][HEAD]["accuracy"] == pytest.approx(1.0)
    assert result["strata"][MID]["accuracy"] == pytest.approx(0.5)
    assert result["strata"][TAIL]["accuracy"] == pytest.approx(0.0)
    assert result["strata"][TAIL]["worst_class_recall"] == pytest.approx(0.0)
    assert result["strata"][TAIL]["zero_recall_class_count"] == 2
    assert result["head_tail_accuracy_gap"] == pytest.approx(1.0)
    assert result["macro_accuracy"] == pytest.approx(3.5 / 6)


def test_long_tail_diagnostics_flags_sparse_strata_as_unstable() -> None:
    stratification = stratify_classes([10, 8, 6, 4, 2, 1], head_fraction=0.5, mid_fraction=0.1667, seed=3)
    targets = torch.tensor([0, 2, 4])
    predictions = torch.tensor([0, 2, 0])
    result = long_tail_diagnostics(
        predictions, targets, num_classes=6, stratification=stratification
    )
    assert result["unstable_strata"] == ["head", "mid", "tail"]
    assert result["head_tail_accuracy_gap"] is None
    assert result["head_tail_macro_gap"] is None
    assert result["strata"][TAIL]["worst_class_recall"] is None
    assert result["strata"][TAIL]["accuracy"] == pytest.approx(0.0)


def test_long_tail_diagnostics_requires_evaluation_support() -> None:
    stratification = stratify_classes([3, 2, 1], head_fraction=0.34, mid_fraction=0.33, seed=1)
    # Head/mid/tail hold class 0, 1 and 2 respectively; evaluating only class 2
    # leaves the head and mid strata without any evaluation support.
    result = long_tail_diagnostics(
        torch.tensor([2]), torch.tensor([2]), num_classes=3, stratification=stratification
    )
    assert result["strata"][HEAD]["accuracy"] is None
    assert result["strata"][MID]["accuracy"] is None
    assert result["strata"][TAIL]["accuracy"] == pytest.approx(1.0)


def test_retention_diagnostics_flags_starved_and_minority_classes() -> None:
    stratification = stratify_classes([10, 8, 6, 4, 2, 1], head_fraction=0.5, mid_fraction=0.1667, seed=3)
    reference = [10, 10, 10, 10, 10, 10]
    retained = [10, 10, 2, 1, 0, 0]
    result = retention_diagnostics(reference, retained, stratification=stratification)
    assert result["starved_class_count"] == 2
    assert result["starved_classes"] == [4, 5]
    assert result["minority_class_count"] == 1
    assert result["min_rate"] == 0.0
    assert result["per_stratum"][TAIL]["starved_class_count"] == 2


def test_retention_gate_fails_on_starvation_and_passes_when_healthy() -> None:
    stratification = stratify_classes([10, 8, 6, 4, 2, 1], head_fraction=0.5, mid_fraction=0.1667, seed=3)
    unhealthy = retention_diagnostics([10] * 6, [10, 10, 2, 1, 0, 0], stratification=stratification)
    verdict = retention_gate(unhealthy, min_class_rate=0.01, min_tail_to_head_ratio=0.05)
    assert verdict["passed"] is False
    assert verdict["checks"]["no_starved_class"]["actual"] == 2
    healthy = retention_diagnostics([10] * 6, [10, 10, 4, 4, 2, 2], stratification=stratification)
    verdict = retention_gate(healthy, min_class_rate=0.01, min_tail_to_head_ratio=0.05)
    assert verdict["passed"] is True
    # Head classes retain 1.0/1.0/0.4 (mean 0.8) and the tail strata retain
    # 0.2 each, so the tail-to-head ratio is 0.25.
    assert verdict["checks"]["tail_to_head_rate_ratio"]["actual"] == pytest.approx(0.25)


def test_retention_diagnostics_rejects_mismatched_lengths() -> None:
    stratification = stratify_classes([3, 2, 1], head_fraction=0.34, mid_fraction=0.33, seed=1)
    with pytest.raises(ValueError, match="equal length"):
        retention_diagnostics([1, 1], [1, 1, 1], stratification=stratification)
