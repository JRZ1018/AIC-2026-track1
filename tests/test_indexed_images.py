from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from PIL import Image

from aictrack1.data.contracts import load_dataset_config
from aictrack1.data.indexed_images import IndexedImageDataset, read_indexed_images
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
