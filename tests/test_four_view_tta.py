from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from aictrack1.data.contracts import TransformConfig
from aictrack1.data.indexed_images import IndexedImage, IndexedImageDataset
from aictrack1.data.transforms import build_split_transforms, _center_crop, _resize_shorter_side
from aictrack1.evaluation.four_view import (
    FOUR_VIEW_RESIZE,
    PairedViewDataset,
    fixed_four_view_logits,
    resolve_four_view_resize,
)
from scripts.diagnose_stage2_tta import fixed_four_view_logits as diagnostic_four_view_logits


def test_fixed_four_view_uses_one_model_equal_float_logits_and_exact_flips() -> None:
    calls = []

    def model(images: torch.Tensor) -> torch.Tensor:
        calls.append(images.clone())
        return torch.stack((images[..., 0].sum(dim=(1, 2)), images[..., -1].sum(dim=(1, 2))), dim=1).half()

    native = torch.tensor([[[[1.0, 2.0, 3.0]]]])
    scaled = torch.tensor([[[[4.0, 5.0, 6.0]]]])
    center, two, four = fixed_four_view_logits(model, native, scaled)

    assert len(calls) == 4
    torch.testing.assert_close(calls[0], native)
    torch.testing.assert_close(calls[1], torch.flip(native, dims=(-1,)))
    torch.testing.assert_close(calls[2], scaled)
    torch.testing.assert_close(calls[3], torch.flip(scaled, dims=(-1,)))
    torch.testing.assert_close(center, torch.tensor([[1.0, 3.0]]))
    torch.testing.assert_close(two, torch.tensor([[2.0, 2.0]]))
    torch.testing.assert_close(four, torch.tensor([[3.5, 3.5]]))
    assert four.dtype == torch.float32


def test_diagnostic_module_reexports_the_same_four_view_function() -> None:
    """The frozen C7 import path must keep resolving to the shared implementation."""

    assert diagnostic_four_view_logits is fixed_four_view_logits


@pytest.mark.parametrize(
    ("resolution", "option", "expected"),
    [
        (224, "scale256", 256),
        (336, "scale384", 384),
        (224, None, None),
        (336, None, None),
    ],
)
def test_resolve_four_view_resize_returns_preregistered_resize(
    resolution: int, option: str | None, expected: int | None
) -> None:
    assert resolve_four_view_resize(resolution, option) == expected


@pytest.mark.parametrize(
    ("resolution", "option"),
    [(224, "scale384"), (336, "scale256")],
)
def test_resolve_four_view_resize_rejects_a_mismatched_resolution(
    resolution: int, option: str
) -> None:
    with pytest.raises(ValueError, match="requires a"):
        resolve_four_view_resize(resolution, option)


def test_resolve_four_view_resize_rejects_an_unknown_option() -> None:
    with pytest.raises(ValueError, match="unknown four-view option"):
        resolve_four_view_resize(336, "scale512")


def test_frozen_options_keep_the_validated_eight_sevenths_short_side_zoom() -> None:
    assert FOUR_VIEW_RESIZE == {"scale256": (224, 256), "scale384": (336, 384)}
    for option, (resolution, resize) in FOUR_VIEW_RESIZE.items():
        assert resize / resolution == pytest.approx(8 / 7), option


def _record(tmp_path, name: str = "sample.jpg") -> IndexedImage:
    path = tmp_path / name
    # Deliberately non-square and larger than the target so the short-side
    # resize and the centre crop are both exercised.
    width, height = 700, 500
    columns = np.arange(width, dtype=np.int16).reshape(1, width)
    rows = np.arange(height, dtype=np.int16).reshape(height, 1)
    pixels = np.stack(
        (
            np.broadcast_to(columns, (height, width)),
            np.broadcast_to(rows, (height, width)),
            (columns + rows) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path, format="JPEG", quality=95)
    return IndexedImage(
        sample_index=0,
        sample_id="stage2-train-000000",
        relative_path=name,
        absolute_path=path,
        source="train",
        partition="validation",
        internal_class_index=7,
        official_class_id="0007",
    )


def test_paired_view_dataset_keeps_the_native_view_identical_to_the_single_view_path(
    tmp_path,
) -> None:
    """Only the extra views may change: the native tensor must be bit-identical."""

    record = _record(tmp_path)
    transforms = TransformConfig(image_size=336, resize_size=336)
    native = build_split_transforms(transforms)["validation"]
    scaled = build_split_transforms(replace(transforms, resize_size=384))["validation"]

    single = IndexedImageDataset([record], native)
    paired = PairedViewDataset([record], native, scaled, required_partition="validation")

    expected = single[0]
    observed = paired[0]
    torch.testing.assert_close(observed["image"], expected["image"])
    assert observed["label"] == expected["label"] == 7
    assert observed["sample_id"] == expected["sample_id"]

    scaled_view = observed["scaled_image"]
    assert scaled_view.shape == (3, 336, 336)
    assert not torch.equal(scaled_view, observed["image"])
    # Deterministic: the same record yields byte-identical views on a rerun.
    torch.testing.assert_close(paired[0]["scaled_image"], scaled_view)


def test_scaled_view_is_the_centre_crop_of_the_preregistered_short_side_resize(
    tmp_path,
) -> None:
    from aictrack1.data.transforms import _normalize, _safe_exif_transpose

    record = _record(tmp_path)
    transforms = TransformConfig(image_size=336, resize_size=336)
    native = build_split_transforms(transforms)["validation"]
    scaled = build_split_transforms(replace(transforms, resize_size=384))["validation"]

    observed = PairedViewDataset([record], native, scaled, required_partition="validation")[0]

    with Image.open(record.absolute_path) as opened:
        reference = _normalize(_center_crop(_resize_shorter_side(_safe_exif_transpose(opened), 384), 336))
    torch.testing.assert_close(observed["scaled_image"], reference)
    # 700x500 -> short side 384 gives round(700 * 384/500) x 384 = 538x384, so the
    # centre crop starts at x=(538-336)//2=101 and the narrow centre crop is exact.
    with Image.open(record.absolute_path) as opened:
        assert _resize_shorter_side(opened, 384).size == (538, 384)
        assert _resize_shorter_side(opened, 336).size == (470, 336)


def test_paired_view_dataset_refuses_a_non_validation_record(tmp_path) -> None:
    record = _record(tmp_path)
    transforms = TransformConfig(image_size=336, resize_size=336)
    native = build_split_transforms(transforms)["validation"]
    scaled = build_split_transforms(replace(transforms, resize_size=384))["validation"]

    with pytest.raises(ValueError, match="train-derived"):
        PairedViewDataset([record], native, scaled, required_partition="train")
    with pytest.raises(ValueError, match="at least one record"):
        PairedViewDataset([], native, scaled, required_partition="validation")


def test_paired_view_dataset_allows_unlabeled_records_without_a_partition_guard(
    tmp_path,
) -> None:
    record = replace(
        _record(tmp_path),
        source="test",
        partition="test",
        internal_class_index=None,
        official_class_id=None,
    )
    transforms = TransformConfig(image_size=336, resize_size=336)
    native = build_split_transforms(transforms)["validation"]
    scaled = build_split_transforms(replace(transforms, resize_size=384))["validation"]

    observed = PairedViewDataset([record], native, scaled, required_partition=None)[0]
    assert observed["label"] == -1
    assert observed["image"].shape == (3, 336, 336)
    assert observed["scaled_image"].shape == (3, 336, 336)
