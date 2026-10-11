from __future__ import annotations

import pytest
import torch

from aictrack1.methods.elr import ELRLoss, ELRState, estimate_elr_state_bytes


def test_elr_state_update_and_restore() -> None:
    state = ELRState(10, 3, max_bytes=10_000)
    loss_fn = ELRLoss(state, beta=0.5, lambda_=1.0)
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]], requires_grad=True)
    loss = loss_fn(logits, torch.tensor([0, 1]), torch.tensor([3, 7]))
    loss.backward()
    assert torch.count_nonzero(state.target[3]) == 3
    saved = state.state_dict()
    restored = ELRState(10, 3, max_bytes=10_000)
    restored.load_state_dict(saved)
    assert torch.equal(state.target, restored.target)


def test_elr_ema_update_matches_definition() -> None:
    state = ELRState(4, 3, max_bytes=1000)
    loss_fn = ELRLoss(state, beta=0.7, lambda_=0.0)
    logits = torch.tensor([[2.0, 0.0, -1.0]])
    probabilities = torch.softmax(logits, dim=1).squeeze(0)
    for _ in range(2):
        loss_fn(logits, torch.tensor([0]), torch.tensor([2]))
    assert torch.allclose(state.target[2], 0.51 * probabilities, atol=1e-6)


def test_elr_unreduced_loss_mean_matches_scalar_loss_and_gradient() -> None:
    logits_mean = torch.tensor([[1.3, 0.1, -0.4], [-0.1, 1.1, 0.2]], requires_grad=True)
    logits_none = logits_mean.detach().clone().requires_grad_(True)
    targets = torch.tensor([0, 1])
    indices = torch.tensor([2, 7])
    mean_fn = ELRLoss(ELRState(10, 3, max_bytes=10_000), beta=0.6, lambda_=1.2)
    none_fn = ELRLoss(ELRState(10, 3, max_bytes=10_000), beta=0.6, lambda_=1.2)

    scalar = mean_fn(logits_mean, targets, indices)
    per_sample = none_fn(logits_none, targets, indices, reduction="none")
    reduced = per_sample.mean()
    scalar.backward()
    reduced.backward()

    assert per_sample.shape == targets.shape
    assert torch.allclose(scalar, reduced, atol=1e-7, rtol=1e-6)
    assert torch.allclose(logits_mean.grad, logits_none.grad, atol=1e-7, rtol=1e-6)


def test_elr_rejects_unknown_reduction() -> None:
    loss_fn = ELRLoss(ELRState(4, 2, max_bytes=1000))
    with pytest.raises(ValueError, match="reduction"):
        loss_fn(torch.randn(2, 2), torch.tensor([0, 1]), torch.tensor([0, 1]), reduction="sum")


def test_elr_memory_guard_and_estimate() -> None:
    assert estimate_elr_state_bytes(100, 20, torch.float16) == 4000
    with pytest.raises(MemoryError):
        ELRState(100, 20, dtype=torch.float32, max_bytes=7999)
    state = ELRState(100, 20, dtype=torch.float16, max_bytes=4000)
    assert state.nbytes == estimate_elr_state_bytes(100, 20, torch.float16) == 4000


def test_elr_rejects_duplicate_indices() -> None:
    state = ELRState(5, 2, max_bytes=1000)
    loss_fn = ELRLoss(state)
    with pytest.raises(ValueError, match="unique"):
        loss_fn(torch.randn(2, 2), torch.tensor([0, 1]), torch.tensor([1, 1]))


def test_elr_state_is_independent_of_loader_order_for_one_epoch() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.5, 0.2], [0.1, 1.4]])
    labels = torch.tensor([0, 1, 0, 1])
    left = ELRState(4, 2, max_bytes=1000)
    right = ELRState(4, 2, max_bytes=1000)
    left_loss = ELRLoss(left, beta=0.6, lambda_=0.0)
    right_loss = ELRLoss(right, beta=0.6, lambda_=0.0)
    for batch in (torch.tensor([0, 1]), torch.tensor([2, 3])):
        left_loss(logits[batch], labels[batch], batch)
    for batch in (torch.tensor([3, 1]), torch.tensor([2, 0])):
        right_loss(logits[batch], labels[batch], batch)
    assert torch.equal(left.target, right.target)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_elr_cuda_amp_with_cpu_state() -> None:
    state = ELRState(20, 4, dtype=torch.float16, device="cpu", max_bytes=10_000)
    loss_fn = ELRLoss(state)
    logits = torch.randn(6, 4, device="cuda", requires_grad=True)
    targets = torch.arange(6, device="cuda") % 4
    indices = torch.arange(6, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = loss_fn(logits, targets, indices)
    loss.backward()
    assert torch.isfinite(loss)
    assert state.target.device.type == "cpu"
