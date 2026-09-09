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


def test_elr_memory_guard_and_estimate() -> None:
    assert estimate_elr_state_bytes(100, 20, torch.float16) == 4000
    with pytest.raises(MemoryError):
        ELRState(100, 20, dtype=torch.float32, max_bytes=7999)


def test_elr_rejects_duplicate_indices() -> None:
    state = ELRState(5, 2, max_bytes=1000)
    loss_fn = ELRLoss(state)
    with pytest.raises(ValueError, match="unique"):
        loss_fn(torch.randn(2, 2), torch.tensor([0, 1]), torch.tensor([1, 1]))


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
