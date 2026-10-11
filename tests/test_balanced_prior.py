from __future__ import annotations

import torch

from scripts.audit_stage2_balanced_prior import (
    _balanced_proxy_indices,
)
from aictrack1.evaluation.balanced_prior import fit_uniform_soft_bias


def test_balanced_proxy_indices_are_deterministic_and_class_balanced() -> None:
    targets = torch.tensor([0, 0, 1, 2, 2, 2])
    first = _balanced_proxy_indices(
        targets, num_classes=3, samples_per_class=4, seed=17
    )
    second = _balanced_proxy_indices(
        targets, num_classes=3, samples_per_class=4, seed=17
    )
    assert torch.equal(first, second)
    assert torch.bincount(targets[first], minlength=3).tolist() == [4, 4, 4]


def test_uniform_soft_bias_reduces_uniform_mass_error() -> None:
    logits = torch.tensor(
        [
            [8.0, 1.0, 0.0],
            [7.0, 2.0, 0.0],
            [6.0, 1.0, 1.0],
            [5.0, 3.0, 0.0],
            [4.0, 2.0, 1.0],
            [3.0, 2.0, 1.0],
        ]
    )
    top2 = torch.topk(logits, k=2, dim=1).values
    scale = float((top2[:, 0] - top2[:, 1]).median().item())
    before = torch.softmax(logits / scale, dim=1).mean(dim=0)
    bias, info = fit_uniform_soft_bias(
        logits, temperature_multiplier=1.0, iterations=60
    )
    after = torch.softmax((logits + bias) / scale, dim=1).mean(dim=0)
    target = torch.full_like(after, 1 / 3)
    assert torch.max(torch.abs(after - target)) < torch.max(torch.abs(before - target))
    assert info["iterations"] <= 60
