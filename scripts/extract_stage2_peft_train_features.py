"""Extract train-only Stage 2 features from a fixed-split PEFT checkpoint.

This is a diagnostic cache for classifier-head experiments. A full-train
checkpoint is rejected because it has seen the fixed validation images.
"""

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

from aictrack1.data.class_mapping import ClassMapping  # noqa: E402
from aictrack1.data.contracts import (  # noqa: E402
    ensure_output_path_is_safe,
    load_dataset_config,
    validate_experiment_runtime_contract,
)
from aictrack1.data.feature_cache import FeatureCacheMetadata, FeatureCacheWriter  # noqa: E402
from aictrack1.data.indexed_images import IndexedImageDataset, read_indexed_images  # noqa: E402
from aictrack1.data.transforms import build_split_transforms  # noqa: E402
from aictrack1.models.clip_vit_b32 import CLIP_MEAN, CLIP_STD, OPENAI_CLIP_COMMIT  # noqa: E402
from aictrack1.models.patch_readout import encode_global_and_patch_mean  # noqa: E402
from aictrack1.models.single_model import (  # noqa: E402
    load_diagnostic_model,
    resolve_relative,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--feature-readout",
        choices=("cls", "cls_patch_mean"),
        default="cls",
        help="Single CLIP forward; optional mean of final patch tokens for train-only diagnosis",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.shard_size < args.batch_size:
        raise ValueError("batch size/workers/shard size configuration is invalid")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")

    config_path = args.experiment_config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    experiment = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = experiment.get("experiment", experiment)
    if experiment.get("competition_stage") != "stage2":
        raise ValueError("only Stage 2 checkpoints are allowed")
    if experiment.get("training_scope") != "fixed_split_validation":
        raise ValueError("full-train checkpoints cannot be used for validation feature selection")
    if checkpoint_path.name != "best.pt":
        raise ValueError("use the fixed-split best.pt checkpoint, not a full-train final.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("training_scope") != "fixed_split_validation":
        raise ValueError("checkpoint must originate from fixed-split training")

    base = config_path.parent
    dataset_path = resolve_relative(base, str(experiment["dataset_config"]))
    manifest_path = resolve_relative(base, str(experiment["manifest"]))
    split_path = resolve_relative(base, str(experiment["split"]))
    dataset_config = load_dataset_config(dataset_path)
    if dataset_config.competition_stage != "stage2":
        raise ValueError("dataset config is not Stage 2")
    validate_experiment_runtime_contract(experiment, dataset_config)
    output = ensure_output_path_is_safe(dataset_config, args.output)
    mapping = ClassMapping.from_csv(dataset_config.class_mapping)
    records = read_indexed_images(
        dataset_config, source="train", manifest_path=manifest_path, split_path=split_path
    )
    validate_experiment_runtime_contract(
        experiment, dataset_config, train_count=len(records), num_classes=len(mapping)
    )
    if args.max_samples is not None:
        records = records[: args.max_samples]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    diagnostic = load_diagnostic_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        class_mapping_digest=mapping.digest(),
        num_classes=len(mapping),
        split_sha256=sha256_file(split_path),
        device=device,
        git_commit=str(checkpoint.get("git_commit") or "unknown_in_checkpoint"),
    )
    if not diagnostic.normalized_features:
        raise ValueError("C2 classifier expects normalized visual features")
    transform = build_split_transforms(dataset_config.transforms)["validation"]
    preprocessing = {
        "resize": 224,
        "center_crop": 224,
        "interpolation": "bicubic",
        "convert_rgb": True,
        "mean": list(CLIP_MEAN),
        "std": list(CLIP_STD),
    }
    if args.feature_readout == "cls_patch_mean":
        preprocessing["feature_readout"] = "normalized_cls_concat_normalized_final_patch_mean"
    metadata = FeatureCacheMetadata(
        source="train",
        dataset_stage=dataset_config.stage,
        dataset_config_sha256=sha256_file(dataset_path),
        manifest_sha256=sha256_file(manifest_path),
        split_sha256=sha256_file(split_path),
        class_mapping_sha256=mapping.digest(),
        clip_weight_sha256=diagnostic.clip_weight_sha256,
        clip_upstream_commit=OPENAI_CLIP_COMMIT,
        preprocessing=preprocessing,
        feature_dim=diagnostic.feature_dim * (2 if args.feature_readout == "cls_patch_mean" else 1),
        feature_dtype="float16",
        normalized=args.feature_readout == "cls",
        expected_record_count=len(records),
        peft_checkpoint_sha256=diagnostic.checkpoint_sha256,
        peft_config_sha256=diagnostic.config_sha256,
    )
    writer = FeatureCacheWriter(output, metadata, resume=args.resume)
    completed = writer.verify_record_prefix(records) if args.resume else 0
    remaining = records[completed:]
    dataset = IndexedImageDataset(
        remaining, transform, allow_truncated_images=dataset_config.allow_truncated_images
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    visual = diagnostic.model.visual.eval()
    pending_features: list[torch.Tensor] = []
    pending_records = []
    cursor = completed
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                if args.feature_readout == "cls_patch_mean":
                    global_features, patch_mean = encode_global_and_patch_mean(visual, images)
                    features = torch.cat(
                        (
                            F.normalize(global_features.float(), dim=-1),
                            F.normalize(patch_mean.float(), dim=-1),
                        ),
                        dim=-1,
                    )
                else:
                    features = F.normalize(visual(images).float(), dim=-1)
            features = features.half().cpu()
            pending_features.append(features)
            pending_records.extend(records[cursor : cursor + len(features)])
            cursor += len(features)
            if len(pending_records) >= args.shard_size:
                writer.append(torch.cat(pending_features), pending_records)
                pending_features, pending_records = [], []
    if pending_records:
        writer.append(torch.cat(pending_features), pending_records)
    complete = writer.finalize()
    stats = {
        "record_count": complete["record_count"],
        "duration_seconds": time.perf_counter() - started,
        "resumed_record_count": completed,
        "peft_checkpoint_sha256": diagnostic.checkpoint_sha256,
        "feature_readout": args.feature_readout,
        "scope": "Stage 2 train manifest only; test was not accessed",
    }
    (output / "extraction_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
