"""Optional higher-resolution forward for the official OpenAI CLIP ViT-B/32.

The topology follows OpenAI/CLIP ``clip/model.py`` at commit
``d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`` (MIT). Only the learned
7x7 visual position grid is interpolated; no weights or modules are replaced.
The original 224 forward remains the exact upstream method.
"""

from __future__ import annotations

import math
from types import MethodType

import torch
import torch.nn.functional as F
from torch import nn


def interpolate_visual_positions(positions: torch.Tensor, grid: int) -> torch.Tensor:
    """Keep the CLS position intact and bicubically resize the spatial grid."""
    if positions.ndim != 2 or grid < 1:
        raise ValueError("expected a 2D position table and a positive grid size")
    source = math.isqrt(positions.shape[0] - 1)
    if source * source + 1 != positions.shape[0] or source != 7:
        raise ValueError("expected the official 7x7 ViT-B/32 visual position grid")
    if grid == source:
        return positions
    cls_position = positions[:1]
    spatial = positions[1:].reshape(1, source, source, positions.shape[1]).permute(0, 3, 1, 2)
    resized = F.interpolate(spatial.float(), size=(grid, grid), mode="bicubic", align_corners=False)
    resized = resized.permute(0, 2, 3, 1).reshape(grid * grid, positions.shape[1])
    return torch.cat((cls_position, resized.to(positions.dtype)), dim=0)


def enable_interpolated_visual_resolution(visual: nn.Module, *, target_resolution: int) -> None:
    """Enable one fixed input size without altering official parameter state.

For 320 and 336 pixels, the unchanged 32-stride patch convolution emits
    a 10x10 token grid. 320 is exactly aligned to the grid; 336 leaves a
    16-pixel right/bottom margin unused. This is an explicit property of the
    original backbone, not a hidden crop or kernel change.
    """
    if target_resolution not in (320, 336):
        raise ValueError("only the preregistered 320 or 336 pixel resolutions are supported")
    existing = getattr(visual, "_aic_interpolated_resolution", None)
    if existing is not None:
        if existing != target_resolution:
            raise ValueError("visual already has a different resolution adapter")
        return
    conv = visual.conv1
    if (
        int(visual.input_resolution) != 224
        or conv.kernel_size != (32, 32)
        or conv.stride != (32, 32)
        or visual.positional_embedding.shape != (50, 768)
        or len(visual.transformer.resblocks) != 12
    ):
        raise ValueError("visual is not the official CLIP ViT-B/32 backbone shape")
    original_forward = visual.forward
    original_keys = tuple(visual.state_dict())

    def forward_with_interpolation(self: nn.Module, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("expected NCHW RGB images")
        if images.shape[-2:] == (224, 224):
            return original_forward(images)
        if images.shape[-2:] != (target_resolution, target_resolution):
            raise ValueError("unexpected visual input resolution")
        tokens = self.conv1(images)
        if tokens.shape[-2:] != (10, 10):
            raise RuntimeError("unchanged ViT-B/32 patch projection did not yield 10x10 tokens")
        tokens = tokens.flatten(2).transpose(1, 2)
        cls = self.class_embedding.to(tokens.dtype).expand(tokens.shape[0], 1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        positions = interpolate_visual_positions(self.positional_embedding, 10)
        tokens = self.ln_pre(tokens + positions.to(tokens.dtype))
        tokens = self.transformer(tokens.transpose(0, 1)).transpose(0, 1)
        pooled = self.ln_post(tokens[:, 0, :])
        return pooled if self.proj is None else pooled @ self.proj

    visual.forward = MethodType(forward_with_interpolation, visual)
    visual._aic_interpolated_resolution = target_resolution
    if tuple(visual.state_dict()) != original_keys:
        raise RuntimeError("resolution adapter unexpectedly changed parameter names")
