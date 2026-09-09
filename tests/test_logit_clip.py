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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_logit_clip_cuda_amp() -> None:
    logits = torch.randn(8, 6, device="cuda", requires_grad=True)
    targets = torch.arange(8, device="cuda") % 6
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = logit_clip_cross_entropy(logits, targets, tau=0.2)
    loss.backward()
    assert torch.isfinite(loss)
