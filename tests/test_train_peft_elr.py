from __future__ import annotations

import torch

from aictrack1.methods.elr import ELRLoss, ELRState
from scripts.train_peft import _base_loss_per_sample, _weighted_base_loss


def test_trainer_elr_returns_per_sample_terms_and_respects_ac_weights() -> None:
    logits = torch.tensor(
        [[1.4, 0.1, -0.2], [-0.2, 1.1, 0.3], [0.2, -0.4, 1.2]],
        requires_grad=True,
    )
    targets = torch.tensor([0, 1, 2])
    sample_indices = torch.tensor([2, 7, 11])
    state = ELRState(12, 3, max_bytes=10_000)
    elr = ELRLoss(state, beta=0.7, lambda_=3.0, eps=1e-6)

    per_sample = _base_loss_per_sample(
        logits,
        targets,
        kind="elr",
        q=0.7,
        label_smoothing=0.0,
        elr_loss=elr,
        sample_indices=sample_indices,
    )
    weights = torch.tensor([0.2, 0.8, 1.0])
    loss = _weighted_base_loss(per_sample, weights)
    loss.backward()

    assert per_sample.shape == targets.shape
    assert torch.isfinite(per_sample).all()
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(torch.nonzero(state.target.sum(dim=1)).flatten(), sample_indices)

