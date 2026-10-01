from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from aictrack1.methods.gce import generalized_cross_entropy
from aictrack1.methods.sce import symmetric_cross_entropy, validate_sce_parameters


@pytest.mark.parametrize("classes", [3, 750])
def test_sce_matches_dense_paper_formula_and_reductions(classes):
    logits = torch.linspace(-2, 2, 2 * classes).reshape(2, classes)
    targets = torch.tensor([0, classes - 1])
    probabilities = logits.softmax(-1)
    label_logs = torch.full_like(logits, -4.0).scatter_(1, targets[:, None], 0.0)
    expected = 0.1 * F.cross_entropy(logits, targets, reduction="none") - (
        probabilities * label_logs
    ).sum(-1)
    actual = symmetric_cross_entropy(logits, targets, reduction="none")
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(symmetric_cross_entropy(logits, targets), expected.mean())
    torch.testing.assert_close(
        symmetric_cross_entropy(logits, targets, reduction="sum"), expected.sum()
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_sce_analytic_gradient_and_fp32_computation(dtype):
    logits = torch.tensor([[0.1, 0.5, -0.2], [1.0, -1.0, 0.0]], dtype=dtype, requires_grad=True)
    targets = torch.tensor([2, 0])
    loss = symmetric_cross_entropy(logits, targets, reduction="sum")
    assert loss.dtype == torch.float32
    loss.backward()
    probability = logits.detach().float().softmax(-1)
    target_probability = probability.gather(1, targets[:, None])
    expected = (0.1 + 4 * target_probability) * (probability - F.one_hot(targets, 3))
    torch.testing.assert_close(logits.grad, expected.to(dtype))


def test_sce_extreme_wrong_logits_keep_finite_nonzero_ce_gradient():
    logits = torch.tensor([[10000.0, -10000.0], [-10000.0, 10000.0]], requires_grad=True)
    loss = symmetric_cross_entropy(logits, torch.tensor([1, 0]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().min() > 0


def test_sce_component_limits_and_confident_correct_example():
    logits = torch.tensor([[1.0, -1.0], [0.1, 0.2]])
    targets = torch.tensor([0, 1])
    torch.testing.assert_close(
        symmetric_cross_entropy(logits, targets, alpha=1, beta=0),
        F.cross_entropy(logits, targets),
    )
    torch.testing.assert_close(
        symmetric_cross_entropy(logits, targets, alpha=0),
        4 * (1 - logits.softmax(-1).gather(1, targets[:, None])).mean(),
    )
    assert symmetric_cross_entropy(torch.tensor([[100.0, -100.0]]), torch.tensor([0])) == 0


@pytest.mark.parametrize("alpha,beta,a", [(-1, 1, -4), (1, -1, -4), (0, 0, -4),
                                         (0.1, 1, 0), (0.1, 1, float("nan")),
                                         (float("inf"), 1, -4)])
def test_sce_rejects_invalid_parameters(alpha, beta, a):
    with pytest.raises(ValueError, match="SCE"):
        validate_sce_parameters(alpha, beta, a)


@pytest.mark.parametrize("logits,targets", [
    (torch.ones(2, 3), torch.tensor([0])),
    (torch.empty(0, 3), torch.empty(0, dtype=torch.long)),
    (torch.ones(2, 1), torch.tensor([0, 0])),
    (torch.ones(2, 3), torch.tensor([0.0, 1.0])),
])
def test_sce_rejects_bad_inputs(logits, targets):
    with pytest.raises(ValueError, match="SCE"):
        symmetric_cross_entropy(logits, targets)


def test_sce_rejects_bad_reduction():
    with pytest.raises(ValueError, match="reduction"):
        symmetric_cross_entropy(torch.ones(1, 3), torch.tensor([0]), reduction="median")


@pytest.fixture(scope="module")
def trainer():
    path = Path(__file__).resolve().parents[1] / "scripts/train_peft.py"
    spec = importlib.util.spec_from_file_location("p5_train_peft", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trainer_preserves_ce_gce_and_integrates_weighted_sce(trainer):
    logits = torch.tensor([[0.1, 1.0, -0.2], [0.0, 0.2, 1.0]], requires_grad=True)
    targets = torch.tensor([0, 2])
    for kind in ["cross_entropy", "gce", "sce"]:
        actual = trainer._base_loss_per_sample(logits, targets, kind=kind, q=0.7, label_smoothing=0.0)
        expected = {
            "cross_entropy": F.cross_entropy(logits, targets, reduction="none"),
            "gce": generalized_cross_entropy(logits, targets, reduction="none"),
            "sce": symmetric_cross_entropy(logits, targets, reduction="none"),
        }[kind]
        assert torch.equal(actual, expected)
    weights = torch.tensor([0.02, 0.8])
    loss = trainer._weighted_base_loss(actual, weights)
    torch.testing.assert_close(loss, (actual * weights).sum() / weights.sum())
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(
        trainer._base_loss_per_sample(logits, targets, kind="cross_entropy", q=0.7, label_smoothing=0.1),
        F.cross_entropy(logits, targets, reduction="none", label_smoothing=0.1),
    )
    with pytest.raises(ValueError, match="separate"):
        trainer._base_loss_per_sample(logits, targets, kind="sce", q=0.7, label_smoothing=0.1)
    with pytest.raises(ValueError, match="unsupported"):
        trainer._base_loss_per_sample(logits, targets, kind="typo", q=0.7, label_smoothing=0.0)
