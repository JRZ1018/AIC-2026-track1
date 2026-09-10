from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from aictrack1.methods.gce import generalized_cross_entropy


def test_gce_matches_formula_and_reductions() -> None:
    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]], dtype=torch.float64)
    targets = torch.tensor([0, 1])
    probability = F.softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
    expected = (1 - probability.pow(0.7)) / 0.7
    actual = generalized_cross_entropy(logits, targets, q=0.7, reduction="none")
    assert torch.allclose(actual.double(), expected, atol=1e-6, rtol=1e-6)
    assert generalized_cross_entropy(logits, targets, q=0.7) == pytest.approx(
        expected.mean().item()
    )


def test_gce_approaches_cross_entropy_for_small_q() -> None:
    logits = torch.tensor([[1.2, -0.4, 0.1], [0.2, 0.7, -1.0]])
    targets = torch.tensor([2, 1])
    gce = generalized_cross_entropy(logits, targets, q=1e-5)
    ce = F.cross_entropy(logits, targets)
    assert gce == pytest.approx(float(ce), rel=2e-3)


def test_gce_has_finite_gradients_for_extreme_logits() -> None:
    logits = torch.tensor([[1.0e4, -1.0e4], [-1.0e4, 1.0e4]], requires_grad=True)
    loss = generalized_cross_entropy(logits, torch.tensor([1, 0]), q=0.7)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("q", [0.0, -0.1, 1.1])
def test_gce_rejects_invalid_q(q: float) -> None:
    with pytest.raises(ValueError, match="q"):
        generalized_cross_entropy(torch.ones(1, 2), torch.tensor([0]), q=q)
