"""Preregistered single-checkpoint two-scale four-view TTA.

The frozen two-scale protocol pairs one trained visual resolution with exactly
one short-side resize for the extra views: ``resize_size / image_size == 8 / 7``
matches the 224 C7 candidate (``256`` over ``224``) and keeps the same relative
zoom when the checkpoint was trained at 336 (``384`` over ``336``).  The mapping
lives here so the train-derived diagnostic path and the final inference path
cannot silently drift apart, and so a mismatched pair fails loudly instead of
running an unregistered protocol.

Every view uses the one frozen checkpoint.  The four float logits are averaged
with equal weight; nothing here selects views, scales or weights from results.
"""

from __future__ import annotations

from typing import Callable

import torch
from PIL import Image, ImageFile
from torch import nn
from torch.utils.data import Dataset

from ..data.indexed_images import IndexedImage


FOUR_VIEW_RESIZE: dict[str, tuple[int, int]] = {
    "scale256": (224, 256),
    "scale384": (336, 384),
}


def resolve_four_view_resize(visual_resolution: int, option: str | None) -> int | None:
    """Return the preregistered short-side resize for one four-view option.

    ``option`` is ``None`` when the caller uses a single-scale protocol, which
    keeps this helper usable by the existing two-view code paths.
    """

    if option is None:
        return None
    try:
        required_resolution, resize = FOUR_VIEW_RESIZE[option]
    except KeyError as exc:
        raise ValueError(f"unknown four-view option {option!r}") from exc
    if int(visual_resolution) != required_resolution:
        raise ValueError(
            f"fixed {option} four-view validation requires a "
            f"{required_resolution}-trained model, got {visual_resolution}"
        )
    return resize


def fixed_four_view_logits(
    model: nn.Module, native: torch.Tensor, scaled: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preregistered equal-logit average of two scales and their exact flips."""

    native_logits = model(native).float()
    native_flip_logits = model(torch.flip(native, dims=(-1,))).float()
    scaled_logits = model(scaled).float()
    scaled_flip_logits = model(torch.flip(scaled, dims=(-1,))).float()
    return native_logits, (native_logits + native_flip_logits) / 2, (
        native_logits + native_flip_logits + scaled_logits + scaled_flip_logits
    ) / 4


class PairedViewDataset(Dataset[dict[str, object]]):
    """Two deterministic views of one image for a preregistered TTA protocol.

    Image loading deliberately mirrors :class:`IndexedImageDataset` exactly
    (truncated-image policy, ``convert("RGB")`` before the transform) so that a
    four-view run differs from its recorded two-view baseline in the extra view
    only.  When ``required_partition`` is set the dataset also refuses any
    record that is not that partition of the Stage 2 train source.
    """

    def __init__(
        self,
        records: list[IndexedImage],
        native_transform: Callable[[Image.Image], torch.Tensor],
        scaled_transform: Callable[[Image.Image], torch.Tensor],
        *,
        required_partition: str | None = None,
        allow_truncated_images: bool = False,
    ) -> None:
        if not records:
            raise ValueError("paired-view dataset requires at least one record")
        if required_partition is not None:
            for record in records:
                if record.source != "train" or record.partition != required_partition:
                    raise ValueError(
                        "paired-view diagnostic requires Stage 2 train-derived "
                        f"{required_partition} records"
                    )
        self.records = records
        self.native_transform = native_transform
        self.scaled_transform = scaled_transform
        self.required_partition = required_partition
        self.allow_truncated_images = allow_truncated_images

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = self.allow_truncated_images
            with Image.open(record.absolute_path) as opened:
                image = opened.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"failed to load audited image {record.absolute_path}") from exc
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous
        return {
            "image": self.native_transform(image),
            "scaled_image": self.scaled_transform(image),
            "label": -1 if record.internal_class_index is None else record.internal_class_index,
            "sample_id": record.sample_id,
        }
