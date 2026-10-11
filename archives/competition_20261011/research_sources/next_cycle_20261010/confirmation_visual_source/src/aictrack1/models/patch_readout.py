"""Read global and patch-mean features from one OpenAI CLIP visual forward.

The transformer hook observes the same tokens used by the official CLS path;
it does not run a second encoder or alter the ViT-B/32 backbone.
"""

from __future__ import annotations

import torch
from torch import nn


def encode_global_and_patch_mean(
    visual: nn.Module, images: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the normal CLIP embedding and projected mean of final patch tokens."""

    transformer = getattr(visual, "transformer", None)
    ln_post = getattr(visual, "ln_post", None)
    projection = getattr(visual, "proj", None)
    if not isinstance(transformer, nn.Module) or not isinstance(ln_post, nn.Module):
        raise TypeError("patch readout requires an OpenAI CLIP vision transformer")
    if not isinstance(projection, torch.Tensor) or projection.ndim != 2:
        raise TypeError("patch readout requires the CLIP visual projection")

    captured: list[torch.Tensor] = []

    def capture_tokens(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("visual transformer did not return a token tensor")
        captured.append(output)

    hook = transformer.register_forward_hook(capture_tokens)
    try:
        global_features = visual(images)
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError("expected exactly one visual transformer forward")
    tokens = captured[0]
    if tokens.ndim != 3 or tokens.shape[0] < 2 or tokens.shape[1] != images.shape[0]:
        raise ValueError("visual transformer token shape is inconsistent with image batch")
    patches = tokens[1:].permute(1, 0, 2)
    patch_features = ln_post(patches) @ projection
    patch_mean = patch_features.mean(dim=1)
    if global_features.shape != patch_mean.shape:
        raise ValueError("global and patch readouts have different feature dimensions")
    return global_features, patch_mean
