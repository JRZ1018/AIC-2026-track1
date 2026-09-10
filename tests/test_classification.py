from __future__ import annotations

import pytest
import torch

from aictrack1.evaluation.classification import classification_diagnostics


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
