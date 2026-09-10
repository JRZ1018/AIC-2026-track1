from __future__ import annotations

import pytest
import torch

from aictrack1.evaluation.classification import calibration_diagnostics, classification_diagnostics


def test_classification_diagnostics_reports_recall_and_confusions() -> None:
    targets = torch.tensor([0, 0, 1, 1, 2, 2])
    predictions = torch.tensor([0, 1, 1, 1, 2, 1])
    result = classification_diagnostics(predictions, targets, num_classes=3)
    assert result["top1"] == pytest.approx(4 / 6)
    assert result["macro_accuracy"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    assert result["predicted_classes"] == 3
    assert result["zero_recall_classes"] == 0
    assert result["top_confusions"][0]["count"] == 1


def test_classification_diagnostics_rejects_out_of_range_classes() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        classification_diagnostics(torch.tensor([3]), torch.tensor([0]), num_classes=3)


def test_calibration_diagnostics_reports_ece_nll_and_logit_norms() -> None:
    logits = torch.tensor([[4.0, 0.0], [0.2, 0.8], [1.0, 2.0]])
    targets = torch.tensor([0, 0, 1])
    result = calibration_diagnostics(logits, targets, num_bins=5)
    assert result["negative_log_likelihood"] > 0
    assert 0 <= result["expected_calibration_error"] <= 1
    assert 0 <= result["mean_confidence"] <= 1
    assert result["logit_l2_norm_quantiles"]["max"] >= result["logit_l2_norm_quantiles"]["median"]
    assert sum(item["count"] for item in result["bins"]) == 3


def test_calibration_diagnostics_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="matching"):
        calibration_diagnostics(torch.ones(2, 3), torch.tensor([0]))
    with pytest.raises(ValueError, match="num_bins"):
        calibration_diagnostics(torch.ones(1, 3), torch.tensor([0]), num_bins=0)
