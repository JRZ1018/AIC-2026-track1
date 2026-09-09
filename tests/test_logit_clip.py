from __future__ import annotations

import pytest
import torch

from aictrack1.methods.logit_clip import logit_clip, logit_clip_cross_entropy


def test_logit_clip_identity_and_bound() -> None:
    logits = torch.tensor([[0.3, 0.4], [30.0, 40.0]], requires_grad=True)
    clipped = logit_clip(logits, tau=0.5)
    assert torch.allclose(clipped[0], logits[0])
    assert torch.linalg.vector_norm(clipped[1], ord=2) <= 2.0 + 1e-6
    loss = logit_clip_cross_entropy(logits, torch.tensor([0, 1]), tau=0.5)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_logit_clip_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError):
        logit_clip(torch.ones(1, 2), tau=0)


def test_logit_clip_preserves_shape_dtype_and_handles_extreme_logits() -> None:
    logits = torch.tensor([[1.0e20, -1.0e20, 3.0]], dtype=torch.float64)
    clipped = logit_clip(logits, tau=0.25)
    assert clipped.shape == logits.shape
    assert clipped.dtype == logits.dtype
    assert clipped.device == logits.device
    assert torch.isfinite(clipped).all()
    assert torch.linalg.vector_norm(clipped, dim=1).item() <= 4.0 + 1e-10


def test_logit_clip_gradient_matches_finite_difference() -> None:
    logits = torch.tensor([[2.5, -0.3, 0.7]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([1])
    loss = logit_clip_cross_entropy(logits, target, tau=0.5)
    loss.backward()
    analytical = float(logits.grad[0, 0])
    delta = 1e-6
    with torch.no_grad():
        plus = logits.detach().clone()
        minus = logits.detach().clone()
        plus[0, 0] += delta
        minus[0, 0] -= delta
        numerical = float(
            (
                logit_clip_cross_entropy(plus, target, tau=0.5)
                - logit_clip_cross_entropy(minus, target, tau=0.5)
            )
            / (2 * delta)
        )
    assert analytical == pytest.approx(numerical, rel=1e-5, abs=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_logit_clip_cuda_amp() -> None:
    logits = torch.randn(8, 6, device="cuda", requires_grad=True)
    targets = torch.arange(8, device="cuda") % 6
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = logit_clip_cross_entropy(logits, targets, tau=0.2)
    loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_logit_clip_cpu_cuda_consistency() -> None:
    logits = torch.randn(8, 6, generator=torch.Generator().manual_seed(8))
    cpu = logit_clip(logits, tau=0.2)
    cuda = logit_clip(logits.cuda(), tau=0.2).cpu()
    assert torch.allclose(cpu, cuda, atol=1e-6, rtol=1e-6)
