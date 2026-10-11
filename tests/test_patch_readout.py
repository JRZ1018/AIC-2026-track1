from __future__ import annotations

import pytest
import torch
from torch import nn

from aictrack1.models.patch_readout import encode_global_and_patch_mean


class TinyVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Identity()
        self.ln_post = nn.LayerNorm(4)
        self.proj = nn.Parameter(torch.arange(1, 13, dtype=torch.float32).reshape(4, 3) / 12)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = torch.stack((images, images + 1, images + 2), dim=0)
        x = self.transformer(x)
        return self.ln_post(x[0]) @ self.proj


def test_patch_readout_preserves_standard_global_path_and_removes_hook() -> None:
    visual = TinyVisual()
    images = torch.tensor([[0.5, -1.0, 2.0, 3.0], [1.0, 3.0, -2.0, 0.0]])
    expected_global = visual(images)
    expected_patches = visual.ln_post(torch.stack((images + 1, images + 2), dim=1))
    expected_mean = (expected_patches @ visual.proj).mean(dim=1)

    global_features, patch_mean = encode_global_and_patch_mean(visual, images)

    torch.testing.assert_close(global_features, expected_global)
    torch.testing.assert_close(patch_mean, expected_mean)
    assert not visual.transformer._forward_hooks


def test_patch_readout_preserves_gradient_to_one_visual_model() -> None:
    visual = TinyVisual()
    images = torch.randn(2, 4, requires_grad=True)
    global_features, patch_mean = encode_global_and_patch_mean(visual, images)
    (global_features.square().sum() + patch_mean.square().sum()).backward()
    assert images.grad is not None and torch.isfinite(images.grad).all()
    assert visual.proj.grad is not None


def test_patch_readout_rejects_non_clip_visual() -> None:
    with pytest.raises(TypeError, match="vision transformer"):
        encode_global_and_patch_mean(nn.Linear(4, 3), torch.randn(2, 4))
