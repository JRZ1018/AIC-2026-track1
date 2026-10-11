"""Explicitly separated train, validation, and test image transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageOps

from .contracts import TransformConfig


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _resize_shorter_side(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    scale = size / min(width, height)
    resized = (max(size, round(width * scale)), max(size, round(height * scale)))
    return image.resize(resized, Image.Resampling.BICUBIC)


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    left = max(0, (width - size) // 2)
    top = max(0, (height - size) // 2)
    return image.crop((left, top, left + size, top + size))


def _ensure_rgb(image: Image.Image) -> Image.Image:
    # Pillow convert("RGB") copies the full pixel buffer even for RGB input.
    return image if image.mode == "RGB" else image.convert("RGB")


def _normalize(image: Image.Image) -> torch.Tensor:
    image = _ensure_rgb(image)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(CLIP_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(CLIP_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def _safe_exif_transpose(image: Image.Image) -> Image.Image:
    """Apply a valid EXIF orientation and ignore only malformed EXIF metadata.

    The decoded pixels remain usable when Pillow rejects a corrupt EXIF/TIFF
    header.  Falling back to the already loaded pixels is deterministic and
    avoids treating metadata damage as permission to skip a training sample.
    """

    try:
        return ImageOps.exif_transpose(image)
    except (OSError, SyntaxError, ValueError):
        fallback = image.copy()
        fallback.info.pop("exif", None)
        return fallback


@dataclass(frozen=True)
class ImageTransform:
    split: str
    config: TransformConfig

    @property
    def stochastic(self) -> bool:
        return self.split == "train"

    def __call__(self, image: Image.Image, *, rng: np.random.Generator | None = None) -> torch.Tensor:
        image = _ensure_rgb(_safe_exif_transpose(image))
        if self.split == "train":
            if rng is None:
                raise ValueError("train transform requires an explicit numpy random generator")
            width, height = image.size
            scale = float(rng.uniform(self.config.train_crop_scale_min, 1.0))
            crop_size = max(1, round(min(width, height) * scale**0.5))
            left = int(rng.integers(0, width - crop_size + 1))
            top = int(rng.integers(0, height - crop_size + 1))
            image = image.crop((left, top, left + crop_size, top + crop_size))
            if rng.random() < self.config.train_horizontal_flip_probability:
                image = ImageOps.mirror(image)
            image = image.resize((self.config.image_size, self.config.image_size), Image.Resampling.BICUBIC)
        elif self.split in {"validation", "test"}:
            image = _center_crop(_resize_shorter_side(image, self.config.resize_size), self.config.image_size)
        else:
            raise ValueError(f"unsupported transform split {self.split!r}")
        return _normalize(image)


def build_split_transforms(config: TransformConfig) -> dict[str, ImageTransform]:
    return {split: ImageTransform(split, config) for split in ("train", "validation", "test")}
