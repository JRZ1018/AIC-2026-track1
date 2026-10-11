import pytest
import torch

from aictrack1.evaluation.hard_capacity import decode_hard_capacity


def _logits_for_predictions(predictions: list[int], classes: int) -> torch.Tensor:
    logits = torch.zeros(len(predictions), classes)
    for row, target in enumerate(predictions):
        logits[row, target] = 1.0
    return logits


def test_hard_capacity_keeps_already_balanced_predictions():
    logits = _logits_for_predictions([0, 0, 1, 1, 2, 2, 3, 3], 4)

    decoded, report = decode_hard_capacity(logits, lower=2, upper=2)

    assert decoded.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert report["floor_phase_moves"] == 0
    assert report["ceiling_phase_moves"] == 0
    assert report["quota_exact"] is True


def test_hard_capacity_runs_floor_then_ceiling_phase():
    logits = _logits_for_predictions([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2], 4)

    decoded, report = decode_hard_capacity(logits, lower=2, upper=3)

    assert torch.bincount(decoded, minlength=4).tolist() == [3, 3, 3, 3]
    assert report["floor_phase_moves"] == 2
    assert report["ceiling_phase_moves"] == 2
    assert report["total_moves"] == 4
    assert report["quota_exact"] is True


def test_hard_capacity_ties_break_by_row_then_class():
    logits = _logits_for_predictions([0, 0, 0, 1, 1, 2], 3)

    decoded, _ = decode_hard_capacity(logits, lower=2, upper=2)

    assert decoded.tolist() == [2, 0, 0, 1, 1, 2]


def test_hard_capacity_rejects_infeasible_or_nonfinite_inputs():
    logits = torch.zeros(6, 3)
    with pytest.raises(ValueError, match="cannot contain"):
        decode_hard_capacity(logits, lower=3, upper=3)
    with pytest.raises(ValueError, match="finite"):
        decode_hard_capacity(torch.tensor([[0.0, float("nan")]]), lower=0, upper=1)
    with pytest.raises(ValueError, match="bounds"):
        decode_hard_capacity(logits, lower=2, upper=1)
