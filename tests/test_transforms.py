from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from aictrack1.data.contracts import TransformConfig
from aictrack1.data.transforms import build_split_transforms


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
