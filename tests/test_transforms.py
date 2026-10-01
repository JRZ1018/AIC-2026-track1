from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps

from aictrack1.data.contracts import TransformConfig
from aictrack1.data.transforms import _normalize, build_split_transforms


def _pattern_image() -> Image.Image:
    array = np.zeros((48, 64, 3), dtype=np.uint8)
    array[:, :32, 0] = 255
    array[10:35, 20:50, 1] = 180
    return Image.fromarray(array, mode="RGB")


def test_validation_and_test_transforms_are_deterministic() -> None:
    transforms = build_split_transforms(TransformConfig(image_size=32, resize_size=40))
    image = _pattern_image()
    validation_1 = transforms["validation"](image)
    validation_2 = transforms["validation"](image)
    test_1 = transforms["test"](image)
    test_2 = transforms["test"](image)
    assert torch.equal(validation_1, validation_2)
    assert torch.equal(test_1, test_2)
    assert torch.equal(validation_1, test_1)
    assert validation_1.shape == (3, 32, 32)
    assert not transforms["validation"].stochastic
    assert not transforms["test"].stochastic


def test_train_transform_requires_explicit_rng_and_replays() -> None:
    transform = build_split_transforms(TransformConfig(image_size=32, resize_size=40))["train"]
    image = _pattern_image()
    with pytest.raises(ValueError, match="explicit"):
        transform(image)
    first = transform(image, rng=np.random.default_rng(9))
    replay = transform(image, rng=np.random.default_rng(9))
    second = transform(image, rng=np.random.default_rng(10))
    assert torch.equal(first, replay)
    assert not torch.equal(first, second)
    assert transform.stochastic


def test_transform_falls_back_to_decoded_pixels_for_malformed_exif() -> None:
    image = _pattern_image()
    image.info["exif"] = b"Exif\x00\x00IIRO\x08\x00\x00\x00"
    transform = build_split_transforms(TransformConfig(image_size=32, resize_size=40))[
        "validation"
    ]

    transformed = transform(image)

    assert transformed.shape == (3, 32, 32)
    assert torch.isfinite(transformed).all()


def test_rgb_transform_skips_redundant_mode_conversions(monkeypatch: pytest.MonkeyPatch) -> None:
    original_convert = Image.Image.convert
    conversions: list[str | None] = []

    def tracked_convert(self: Image.Image, mode: str | None = None, *args: object, **kwargs: object) -> Image.Image:
        conversions.append(mode)
        return original_convert(self, mode, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "convert", tracked_convert)
    transforms = build_split_transforms(TransformConfig(image_size=32, resize_size=40))
    rgb = _pattern_image()
    transforms["train"](rgb, rng=np.random.default_rng(9))
    transforms["validation"](rgb)
    transforms["test"](rgb)
    _normalize(rgb)
    assert conversions == []


def test_non_rgb_transform_retains_conversion_and_pixel_values(monkeypatch: pytest.MonkeyPatch) -> None:
    gray = _pattern_image().convert("L")
    original_convert = Image.Image.convert
    conversions: list[str | None] = []

    def tracked_convert(self: Image.Image, mode: str | None = None, *args: object, **kwargs: object) -> Image.Image:
        conversions.append(mode)
        return original_convert(self, mode, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "convert", tracked_convert)
    transform = build_split_transforms(TransformConfig(image_size=32, resize_size=40))["validation"]
    result = transform(gray)
    reference = transform(gray.convert("RGB"))
    assert torch.equal(result, reference)
    assert conversions.count("RGB") == 2  # one in the transform, one explicit reference conversion


def test_exif_orientation_is_preserved_without_redundant_rgb_conversion() -> None:
    image = _pattern_image()
    exif = image.getexif()
    exif[274] = 6  # rotate 90 degrees clockwise
    image.info["exif"] = exif.tobytes()
    transform = build_split_transforms(TransformConfig(image_size=32, resize_size=40))["validation"]
    oriented = ImageOps.exif_transpose(image)
    assert oriented.size == (48, 64)
    assert torch.equal(transform(image), transform(oriented))
