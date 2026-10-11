"""Small CPU checks for the official ViT-B/32 position interpolation adapter."""

import pytest
import torch

from aictrack1.models.clip_resolution import (
    enable_interpolated_visual_resolution,
    interpolate_visual_positions,
)


def test_position_grid_preserves_cls_and_source() -> None:
    positions = torch.arange(50 * 768, dtype=torch.float32).reshape(50, 768)
    assert interpolate_visual_positions(positions, 7) is positions
    enlarged = interpolate_visual_positions(positions, 10)
    assert enlarged.shape == (101, 768)
    torch.testing.assert_close(enlarged[0], positions[0], rtol=0, atol=0)


def test_visual_adapter_preserves_224_and_state_dict() -> None:
    from clip.model import VisionTransformer

    torch.manual_seed(123)
    visual = VisionTransformer(224, 32, 768, 12, 12, 512).eval()
    images = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        original = visual(images)
    state = {name: value.clone() for name, value in visual.state_dict().items()}
    enable_interpolated_visual_resolution(visual, target_resolution=336)
    with torch.no_grad():
        adapted_224 = visual(images)
        adapted_336 = visual(torch.randn(1, 3, 336, 336))
    torch.testing.assert_close(adapted_224, original, rtol=0, atol=0)
    assert adapted_336.shape == (1, 512)
    assert torch.isfinite(adapted_336).all()
    assert tuple(visual.state_dict()) == tuple(state)
    for name, value in visual.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="unexpected visual input resolution"):
        visual(torch.randn(1, 3, 320, 320))


def test_visual_adapter_supports_exact_320_grid() -> None:
    from clip.model import VisionTransformer

    visual = VisionTransformer(224, 32, 768, 12, 12, 512).eval()
    enable_interpolated_visual_resolution(visual, target_resolution=320)
    with torch.no_grad():
        output = visual(torch.randn(1, 3, 320, 320))
    assert output.shape == (1, 512)
    assert torch.isfinite(output).all()
