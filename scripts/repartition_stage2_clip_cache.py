"""Rebind audited Stage 2 train-only OpenAI CLIP features to a new split.

Visual features are unchanged: this only writes a new provenance-bound cache
whose train/validation partition assignments come from the supplied split.
Competition test data are never accepted or read.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.class_mapping import ClassMapping  # noqa: E402
from aictrack1.data.contracts import ensure_output_path_is_safe, load_dataset_config  # noqa: E402
from aictrack1.data.feature_cache import FeatureCacheWriter, load_feature_cache  # noqa: E402
from aictrack1.data.indexed_images import read_indexed_images  # noqa: E402
from aictrack1.data.manifest import sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache", required=True, type=Path)
    parser.add_argument("--source-dataset-config", required=True, type=Path)
    parser.add_argument("--source-split", required=True, type=Path)
    parser.add_argument("--target-dataset-config", required=True, type=Path)
    parser.add_argument("--target-split", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.shard_size < 1:
        raise ValueError("shard-size must be positive")

    source_config_path = args.source_dataset_config.resolve()
    target_config_path = args.target_dataset_config.resolve()
    source_split_path = args.source_split.resolve()
    target_split_path = args.target_split.resolve()
    manifest_path = args.manifest.resolve()
    source_config = load_dataset_config(source_config_path)
    target_config = load_dataset_config(target_config_path)
    if source_config.competition_stage != "stage2" or target_config.competition_stage != "stage2":
        raise ValueError("only Stage 2 train data are allowed")
    if source_config.sources["train"].root != target_config.sources["train"].root:
        raise ValueError("source and target train roots differ")
    if source_config.transforms != target_config.transforms:
        raise ValueError("source and target image preprocessing differs")
    if source_config.class_mapping != target_config.class_mapping:
        raise ValueError("source and target class mapping paths differ")
    if source_config.expected_train_count != target_config.expected_train_count:
        raise ValueError("source and target train counts differ")
    source_split_sha = sha256_file(source_split_path)
    target_split_sha = sha256_file(target_split_path)
    if source_split_sha == target_split_sha:
        raise ValueError("target split must differ from source split")
    mapping = ClassMapping.from_csv(target_config.class_mapping)
    cache = load_feature_cache(args.source_cache.resolve())
    metadata = cache.metadata
    if metadata.source != "train" or metadata.peft_checkpoint_sha256 is not None:
        raise ValueError("source must be train-only original CLIP features, not PEFT or test")
    if metadata.dataset_config_sha256 != sha256_file(source_config_path):
        raise ValueError("source cache dataset config hash mismatch")
    if metadata.split_sha256 != source_split_sha:
        raise ValueError("source cache split hash mismatch")
    if metadata.manifest_sha256 != sha256_file(manifest_path):
        raise ValueError("source cache manifest hash mismatch")
    if metadata.class_mapping_sha256 != mapping.digest():
        raise ValueError("source cache class mapping hash mismatch")
    if metadata.expected_record_count != target_config.expected_train_count:
        raise ValueError("source cache train count mismatch")

    records = read_indexed_images(
        target_config,
        source="train",
        manifest_path=manifest_path,
        split_path=target_split_path,
    )
    if len(records) != metadata.expected_record_count:
        raise ValueError("new split does not cover the complete train manifest")
    for index, record in enumerate(records):
        if (
            cache.sample_ids[index] != record.sample_id
            or cache.relative_paths[index] != record.relative_path
            or int(cache.sample_indices[index]) != record.sample_index
            or int(cache.labels[index]) != record.internal_class_index
        ):
            raise ValueError(f"source cache identity or label mismatch at row {index}")
    partitions = {record.partition for record in records}
    if partitions != {"train", "validation"}:
        raise ValueError(f"new split has unexpected partitions: {partitions}")

    output = ensure_output_path_is_safe(target_config, args.output)
    if output == args.source_cache.resolve():
        raise ValueError("output must not overwrite source cache")
    new_metadata = replace(
        metadata,
        dataset_config_sha256=sha256_file(target_config_path),
        split_sha256=target_split_sha,
    )
    writer = FeatureCacheWriter(output, new_metadata, resume=args.resume)
    completed = writer.verify_record_prefix(records) if args.resume else 0
    for start in range(completed, len(records), args.shard_size):
        stop = min(start + args.shard_size, len(records))
        # A contiguous slice still shares the full cache storage. torch.save
        # serializes that backing storage unless the slice is cloned first.
        writer.append(cache.features[start:stop].clone(), records[start:stop])
    completion = writer.finalize()
    print(
        json.dumps(
            {
                "scope": "Stage 2 train only; original OpenAI CLIP features unchanged",
                "source_cache": str(args.source_cache.resolve()),
                "output": str(output),
                "source_split_sha256": source_split_sha,
                "target_split_sha256": target_split_sha,
                "record_count": completion["record_count"],
                "resumed_records": completed,
                "shards": len(completion["shards"]),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
