from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from aictrack1.methods.nce_rce import (
    normalized_cross_entropy_plus_reverse_cross_entropy,
    validate_nce_rce_parameters,
)


def test_nce_rce_matches_dense_formula_and_reductions():
    logits = torch.tensor([[0.2, 1.1, -0.4], [1.0, -0.7, 0.3]], dtype=torch.float32)
    targets = torch.tensor([1, 0], dtype=torch.long)
    logp = F.log_softmax(logits, dim=-1)
    nce = -logp.gather(1, targets[:, None]).squeeze(1) / -logp.sum(-1)
    probabilities = logp.exp()
    log_target = torch.full_like(logits, -4.0).scatter_(1, targets[:, None], 0.0)
    rce = -(probabilities * log_target).sum(-1)
    expected = 750.0 * nce + 0.1 * rce
    actual = normalized_cross_entropy_plus_reverse_cross_entropy(logits, targets, reduction="none")
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(normalized_cross_entropy_plus_reverse_cross_entropy(logits, targets), expected.mean())
    torch.testing.assert_close(normalized_cross_entropy_plus_reverse_cross_entropy(logits, targets, reduction="sum"), expected.sum())


def test_nce_rce_fp16_input_uses_finite_fp32_calculation_and_gradient():
    logits = torch.tensor([[0.1, 0.5, -0.2], [1.0, -1.0, 0.0]], dtype=torch.float16, requires_grad=True)
    targets = torch.tensor([2, 0], dtype=torch.long)
    loss = normalized_cross_entropy_plus_reverse_cross_entropy(logits, targets)
    assert loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_nce_rce_extreme_logits_remain_finite():
    logits = torch.tensor([[10000.0, -10000.0, -5000.0], [-10000.0, 10000.0, 0.0]], requires_grad=True)
    loss = normalized_cross_entropy_plus_reverse_cross_entropy(logits, torch.tensor([1, 0], dtype=torch.long))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("alpha,beta,a", [(0, 1, -4), (1, 0, -4), (1, 1, 0), (float("nan"), 1, -4)])
def test_nce_rce_rejects_invalid_parameters(alpha, beta, a):
    with pytest.raises(ValueError, match=r"NCE\+RCE"):
        validate_nce_rce_parameters(alpha, beta, a)


def test_nce_rce_rejects_bad_shapes_and_labels():
    with pytest.raises(ValueError, match="expects"):
        normalized_cross_entropy_plus_reverse_cross_entropy(torch.ones(2, 3), torch.tensor([0]))
    with pytest.raises(ValueError, match="class range"):
        normalized_cross_entropy_plus_reverse_cross_entropy(torch.ones(1, 3), torch.tensor([3]))
