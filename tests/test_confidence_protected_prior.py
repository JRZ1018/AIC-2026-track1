from __future__ import annotations

import pytest
import torch

from aictrack1.evaluation.confidence_protected_prior import (
    confidence_protected_predictions,
)


def test_protects_only_large_margin_predictions() -> None:
    logits = torch.tensor([[5.0, 1.0], [1.5, 1.0], [2.0, 0.0]])
    bias = torch.tensor([0.0, 6.0])
    p7, p20, protected = confidence_protected_predictions(
        logits, bias, median_margin=1.0
    )
    assert protected.tolist() == [True, False, True]
    assert p7.tolist() == [1, 1, 1]
    assert p20.tolist() == [0, 1, 0]


@pytest.mark.parametrize("median", [0.0, float("nan"), float("inf")])
def test_refuses_invalid_margin(median: float) -> None:
    with pytest.raises(ValueError, match="median_margin"):
        confidence_protected_predictions(
            torch.tensor([[1.0, 0.0]]), torch.zeros(2), median_margin=median
        )


def test_refuses_parameter_retuning() -> None:
    with pytest.raises(ValueError, match="multiple"):
        confidence_protected_predictions(
            torch.tensor([[1.0, 0.0]]), torch.zeros(2),
            median_margin=1.0, margin_multiple=1.5,
        )
