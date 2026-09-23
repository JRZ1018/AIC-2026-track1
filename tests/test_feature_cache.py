from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from aictrack1.data.feature_cache import (
    FeatureCacheMetadata,
    FeatureCacheWriter,
    load_feature_cache,
    select_feature_partition,
)
from aictrack1.data.indexed_images import IndexedImage


def _metadata(count: int = 4) -> FeatureCacheMetadata:
    return FeatureCacheMetadata(
        source="train",
        dataset_stage="preliminary",
        dataset_config_sha256="a" * 64,
        manifest_sha256="b" * 64,
        split_sha256="c" * 64,
        class_mapping_sha256="d" * 64,
        clip_weight_sha256="e" * 64,
        clip_upstream_commit="f" * 40,
        preprocessing={"resize": 224},
        feature_dim=3,
        feature_dtype="float16",
        normalized=True,
        expected_record_count=count,
    )


def _records(tmp_path) -> list[IndexedImage]:
    result = []
    for index, partition in enumerate(("train", "validation", "train", "validation")):
        path = tmp_path / f"{index}.jpg"
        path.write_bytes(b"unused")
        result.append(
            IndexedImage(index, f"id-{index}", path.name, path, "train", partition, index % 2, f"{index % 2:04d}")
        )
    return result


def test_feature_cache_round_trip_and_resume_prefix(tmp_path) -> None:
    records = _records(tmp_path)
    root = tmp_path / "cache"
    features = torch.nn.functional.normalize(torch.arange(1, 13).reshape(4, 3).float(), dim=1).half()
    writer = FeatureCacheWriter(root, _metadata())
    writer.append(features[:2], records[:2])
    resumed = FeatureCacheWriter(root, _metadata(), resume=True)
    assert resumed.verify_record_prefix(records) == 2
    resumed.append(features[2:], records[2:])
    complete = resumed.finalize()
    assert complete["record_count"] == 4

    loaded = load_feature_cache(root)
    train_features, train_labels = select_feature_partition(loaded, "train")
    assert train_features.shape == (2, 3)
    assert train_labels.tolist() == [0, 0]
    assert loaded.sample_ids == ("id-0", "id-1", "id-2", "id-3")


def test_feature_cache_refuses_metadata_or_order_mismatch(tmp_path) -> None:
    records = _records(tmp_path)
    root = tmp_path / "cache"
    writer = FeatureCacheWriter(root, _metadata())
    features = torch.nn.functional.normalize(torch.ones(2, 3), dim=1).half()
    writer.append(features, records[:2])
    with pytest.raises(ValueError, match="metadata mismatch"):
        FeatureCacheWriter(root, replace(_metadata(), feature_dim=4), resume=True)
    resumed = FeatureCacheWriter(root, _metadata(), resume=True)
    with pytest.raises(ValueError, match="record order mismatch"):
        resumed.verify_record_prefix(list(reversed(records)))


def test_peft_feature_cache_binds_exact_checkpoint(tmp_path) -> None:
    records = _records(tmp_path)
    root = tmp_path / "peft-cache"
    metadata = replace(
        _metadata(count=2),
        peft_checkpoint_sha256="1" * 64,
        peft_config_sha256="2" * 64,
    )
    writer = FeatureCacheWriter(root, metadata)
    features = torch.nn.functional.normalize(torch.ones(2, 3), dim=1).half()
    writer.append(features, records[:2])
    with pytest.raises(ValueError, match="metadata mismatch"):
        FeatureCacheWriter(
            root,
            replace(metadata, peft_checkpoint_sha256="3" * 64),
            resume=True,
        )
    writer.finalize()
    loaded = load_feature_cache(root)
    assert loaded.metadata.peft_checkpoint_sha256 == "1" * 64
