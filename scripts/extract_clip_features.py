"""Extract a resumable frozen OpenAI CLIP ViT-B/32 feature cache."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.class_mapping import ClassMapping
from aictrack1.data.contracts import ensure_output_path_is_safe, load_dataset_config
from aictrack1.data.feature_cache import FeatureCacheMetadata, FeatureCacheWriter
from aictrack1.data.indexed_images import IndexedImageDataset, read_indexed_images
from aictrack1.data.manifest import sha256_file
from aictrack1.models.clip_vit_b32 import load_clip_vit_b32


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source", required=True, choices=("train", "test"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--expected-weight-sha256")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--feature-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.shard_size < args.batch_size:
        raise ValueError("batch size/workers/shard size configuration is invalid")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max-samples must be positive")
    if args.source == "train" and args.split is None:
        raise ValueError("--split is mandatory for train feature extraction")
    if args.source == "test" and args.split is not None:
        raise ValueError("--split is forbidden for test feature extraction")
    if args.model_config is not None:
        if args.weights is not None or args.expected_weight_sha256 is not None:
            raise ValueError("use either --model-config or explicit weight arguments, not both")
        model_config_path = args.model_config.resolve()
        model_raw = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
        model_raw = model_raw.get("model", model_raw)
        if model_raw.get("provider") != "OpenAI" or model_raw.get("name") != "ViT-B/32":
            raise ValueError("model config must declare the OpenAI ViT-B/32 backbone")
        weight_value = Path(str(model_raw["weight_path"]))
        weight_path = (
            weight_value if weight_value.is_absolute() else model_config_path.parent / weight_value
        ).resolve()
        expected_weight_sha256 = str(model_raw["expected_weight_sha256"])
    else:
        if args.weights is None or args.expected_weight_sha256 is None:
            raise ValueError("explicit loading requires --weights and --expected-weight-sha256")
        weight_path = args.weights
        expected_weight_sha256 = args.expected_weight_sha256
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    dataset_config = load_dataset_config(args.config)
    output = ensure_output_path_is_safe(dataset_config, args.output)
    records = read_indexed_images(
        dataset_config,
        source=args.source,
        manifest_path=args.manifest,
        split_path=args.split,
    )
    if args.max_samples is not None:
        records = records[: args.max_samples]

    model, preprocess, clip_metadata = load_clip_vit_b32(
        weight_path, expected_weight_sha256, device
    )
    metadata = FeatureCacheMetadata(
        source=args.source,
        dataset_stage=dataset_config.stage,
        dataset_config_sha256=sha256_file(dataset_config.config_path),
        manifest_sha256=sha256_file(args.manifest),
        split_sha256=sha256_file(args.split) if args.split is not None else None,
        class_mapping_sha256=ClassMapping.from_csv(dataset_config.class_mapping).digest(),
        clip_weight_sha256=clip_metadata.weight_sha256,
        clip_upstream_commit=clip_metadata.upstream_commit,
        preprocessing=clip_metadata.preprocessing,
        feature_dim=clip_metadata.feature_dim,
        feature_dtype=args.feature_dtype,
        normalized=True,
        expected_record_count=len(records),
    )
    writer = FeatureCacheWriter(output, metadata, resume=args.resume)
    completed = writer.verify_record_prefix(records) if args.resume else 0
    remaining = records[completed:]
    dataset = IndexedImageDataset(
        remaining,
        preprocess,
        allow_truncated_images=dataset_config.allow_truncated_images,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    storage_dtype = torch.float16 if args.feature_dtype == "float16" else torch.float32
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    cursor = completed
    pending_features: list[torch.Tensor] = []
    pending_records = []

    def flush() -> None:
        nonlocal pending_features, pending_records
        if pending_records:
            writer.append(torch.cat(pending_features, dim=0), pending_records)
            pending_features = []
            pending_records = []

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                features = model.encode_image(images)
            features = F.normalize(features.float(), dim=1).to(dtype=storage_dtype).cpu()
            batch_records = records[cursor : cursor + len(features)]
            pending_features.append(features)
            pending_records.extend(batch_records)
            cursor += len(features)
            if len(pending_records) >= args.shard_size:
                flush()
    flush()
    complete = writer.finalize()
    duration = time.perf_counter() - started
    stats = {
        "record_count": complete["record_count"],
        "duration_seconds": duration,
        "samples_per_second": 0.0 if not remaining else len(remaining) / duration,
        "resumed_record_count": completed,
        "peak_vram_mib": (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
        ),
        "clip": clip_metadata.to_dict(),
    }
    (output / "extraction_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
