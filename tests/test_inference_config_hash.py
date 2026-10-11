from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.infer_and_build_submission import (
    SUPPORTED_FINAL_PEFT_MODES,
    _fixed_horizontal_flip_logits,
    _text_sha256_variants,
    _validate_resolution_contract,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_text_sha256_variants_accepts_only_line_ending_changes(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_bytes(b"experiment:\n  seed: 1\n")

    variants = _text_sha256_variants(config)

    assert _sha256(b"experiment:\n  seed: 1\n") in variants
    assert _sha256(b"experiment:\r\n  seed: 1\r\n") in variants
    assert _sha256(b"experiment:\n  seed: 2\n") not in variants


def test_final_inference_accepts_supported_single_model_peft_scopes() -> None:
    assert SUPPORTED_FINAL_PEFT_MODES == {
        "last_n_blocks",
        "lora_last_attention",
        "lora_last_n_attention",
    }


def test_final_inference_preserves_legacy_224_contract() -> None:
    dataset = SimpleNamespace(transforms=SimpleNamespace(image_size=224))
    assert _validate_resolution_contract({}, dataset, {}) == 224


def test_final_inference_requires_matching_336_checkpoint_and_transform() -> None:
    dataset = SimpleNamespace(transforms=SimpleNamespace(image_size=336))
    experiment = {
        "visual_input_resolution": 336,
        "position_interpolation": "bicubic_7x7_to_10x10",
    }
    checkpoint = dict(experiment)
    assert _validate_resolution_contract(experiment, dataset, checkpoint) == 336
    with pytest.raises(ValueError, match="visual resolutions differ"):
        _validate_resolution_contract(experiment, dataset, {**checkpoint, "visual_input_resolution": 224})
    with pytest.raises(ValueError, match="position interpolation differ"):
        _validate_resolution_contract(experiment, dataset, {**checkpoint, "position_interpolation": None})


def test_fixed_flip_tta_uses_same_model_and_equal_float_logits() -> None:
    calls = []

    def model(images: torch.Tensor) -> torch.Tensor:
        calls.append(images.clone())
        return torch.stack((images[..., 0].sum(dim=(1, 2)), images[..., -1].sum(dim=(1, 2))), dim=1).half()

    images = torch.tensor([[[[1.0, 2.0, 3.0]]]])
    logits = _fixed_horizontal_flip_logits(model, images)

    assert len(calls) == 2
    torch.testing.assert_close(calls[0], images)
    torch.testing.assert_close(calls[1], torch.flip(images, dims=(-1,)))
    torch.testing.assert_close(logits, torch.tensor([[2.0, 2.0]]))
    assert logits.dtype == torch.float32
