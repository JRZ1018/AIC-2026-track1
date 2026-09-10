from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from PIL import Image

from aictrack1.data.contracts import load_dataset_config
from aictrack1.data.indexed_images import IndexedImage, IndexedImageDataset, read_indexed_images
from aictrack1.data.transforms import build_split_transforms
from aictrack1.data.manifest import build_manifest, write_manifest


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_indexed_images_follow_manifest_and_fixed_split(synthetic_dataset) -> None:
    dataset_root, config_path = synthetic_dataset
    config = load_dataset_config(config_path)
    records = [record for record in build_manifest(config, "train") if record.decode_status == "ok"][:4]
    subset_manifest = dataset_root.parent / "subset.csv"
    write_manifest(records, subset_manifest)
    split = dataset_root.parent / "split.csv"
    _write_csv(
        split,
        ["sample_id", "split"],
        [
            {"sample_id": record.sample_id, "split": "train" if index < 3 else "validation"}
            for index, record in enumerate(records)
        ],
    )
    indexed = read_indexed_images(config, source="train", manifest_path=subset_manifest, split_path=split)
    assert [record.partition for record in indexed] == ["train", "train", "train", "validation"]
    dataset = IndexedImageDataset(indexed, lambda image: torch.tensor(image.size))
    assert dataset[0]["label"] >= 0


def test_test_source_rejects_split_assignment(synthetic_dataset) -> None:
    _, config_path = synthetic_dataset
    config = load_dataset_config(config_path)
    with pytest.raises(ValueError, match="not declared"):
        read_indexed_images(config, source="test", manifest_path="missing.csv")


def test_stochastic_dataset_transform_is_epoch_seeded(synthetic_dataset) -> None:
    dataset_root, config_path = synthetic_dataset
    config = load_dataset_config(config_path)
    record = next(record for record in build_manifest(config, "train") if record.decode_status == "ok")
    records = [
        IndexedImage(
            sample_index=0,
            sample_id=record.sample_id,
            relative_path=record.relative_path,
            absolute_path=config.sources["train"].root / record.relative_path,
            source="train",
            partition="train",
            internal_class_index=record.internal_class_index,
            official_class_id=record.official_class_id,
        )
    ]
    transform = build_split_transforms(config.transforms)["train"]
    dataset = IndexedImageDataset(records, transform, stochastic_seed=77)
    first = dataset[0]["image"]
    repeated = dataset[0]["image"]
    assert torch.equal(first, repeated)
    dataset.set_epoch(1)
    changed = dataset[0]["image"]
    assert not torch.equal(first, changed)


def test_stochastic_dataset_transform_requires_seed(synthetic_dataset) -> None:
    dataset_root, config_path = synthetic_dataset
    config = load_dataset_config(config_path)
    record = next(record for record in build_manifest(config, "train") if record.decode_status == "ok")
    records = [
        IndexedImage(
            sample_index=0,
            sample_id=record.sample_id,
            relative_path=record.relative_path,
            absolute_path=config.sources["train"].root / record.relative_path,
            source="train",
            partition="train",
            internal_class_index=record.internal_class_index,
            official_class_id=record.official_class_id,
        )
    ]
    dataset = IndexedImageDataset(records, build_split_transforms(config.transforms)["train"])
    with pytest.raises(ValueError, match="explicit dataset seed"):
        dataset[0]
