"""Run test inference for one locked frozen OpenAI CLIP classifier head."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.class_mapping import ClassMapping
from aictrack1.data.contracts import ensure_output_path_is_safe, load_dataset_config
from aictrack1.data.indexed_images import IndexedImageDataset, read_indexed_images
from aictrack1.models.clip_vit_b32 import load_clip_vit_b32
from aictrack1.models.heads import (
    FrozenFeatureImageClassifier,
    load_classifier_head_checkpoint,
)
from aictrack1.submission import SubmissionSpec, build_submission_package


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_settings(path: Path) -> tuple[Path, str]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = document.get("model", document)
    if raw.get("provider") != "OpenAI" or raw.get("name") != "ViT-B/32":
        raise ValueError("model config must declare OpenAI ViT-B/32")
    if raw.get("runtime_downloads_allowed") is not False:
        raise ValueError("runtime downloads must be disabled")
    value = Path(str(raw["weight_path"]))
    weight_path = (value if value.is_absolute() else path.parent / value).resolve()
    return weight_path, str(raw["expected_weight_sha256"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument("--head-checkpoint", required=True, type=Path)
    parser.add_argument("--expected-head-sha256", required=True)
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--test-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")

    experiment_path = args.experiment_config.resolve()
    experiment_document = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    experiment = experiment_document.get("experiment", experiment_document)
    if "peft" in experiment:
        raise ValueError("frozen-head inference refuses PEFT experiment configs")
    if str(experiment.get("head", {}).get("kind")) not in {"linear", "cosine"}:
        raise ValueError("experiment must declare a supported frozen head")

    head_path = args.head_checkpoint.resolve()
    head_sha256 = _sha256(head_path)
    if head_sha256.lower() != args.expected_head_sha256.lower():
        raise ValueError("frozen head checkpoint SHA-256 mismatch")
    split_path = args.split.resolve()
    test_manifest = args.test_manifest.resolve()
    dataset_path = args.dataset_config.resolve()
    model_path = args.model_config.resolve()
    dataset_config = load_dataset_config(dataset_path)
    output = ensure_output_path_is_safe(dataset_config, args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"submission output directory is not empty: {output}")

    mapping = ClassMapping.from_csv(dataset_config.class_mapping)
    if int(experiment["num_classes"]) != len(mapping):
        raise ValueError("experiment class count differs from the official mapping")
    test_records = read_indexed_images(
        dataset_config,
        source="test",
        manifest_path=test_manifest,
    )
    filenames = [Path(record.relative_path).name for record in test_records]
    if len(set(filenames)) != len(filenames):
        raise ValueError("test manifest does not map to unique submission basenames")
    if any(Path(record.relative_path).name != record.relative_path for record in test_records):
        raise ValueError("competition test manifest must contain root-level image filenames")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    weight_path, expected_weight_sha256 = _model_settings(model_path)
    clip_model, preprocess, clip_metadata = load_clip_vit_b32(
        weight_path, expected_weight_sha256, device
    )
    visual = clip_model.visual.float()
    del clip_model
    head, checkpoint = load_classifier_head_checkpoint(
        head_path,
        feature_dim=clip_metadata.feature_dim,
        num_classes=len(mapping),
        expected_clip_weight_sha256=clip_metadata.weight_sha256,
        expected_class_mapping_sha256=mapping.digest(),
        expected_split_sha256=_sha256(split_path),
    )
    metadata = checkpoint["feature_cache_metadata"]
    if metadata.get("dataset_config_sha256") != _sha256(dataset_path):
        raise ValueError("frozen head dataset config provenance mismatch")
    if metadata.get("preprocessing") != clip_metadata.preprocessing:
        raise ValueError("frozen head preprocessing provenance mismatch")
    for parameter in visual.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    model = FrozenFeatureImageClassifier(visual, head).to(device).eval()

    test_dataset = IndexedImageDataset(
        test_records,
        preprocess,
        allow_truncated_images=dataset_config.allow_truncated_images,
    )
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    predicted_indices: list[int] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
            predicted_indices.extend(int(value) for value in logits.argmax(dim=1).cpu())
    if len(predicted_indices) != len(filenames):
        raise RuntimeError("test inference output count does not match manifest")
    inference_seconds = time.perf_counter() - started

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    package = build_submission_package(
        dict(zip(filenames, predicted_indices, strict=True)),
        filenames,
        mapping,
        output,
        run_metadata={
            "run_id": str(experiment["run_id"]),
            "checkpoint_sha256": head_sha256,
            "config_hash": _sha256(experiment_path),
            "data_hash": _sha256(test_manifest),
            "git_commit": git_commit,
            "inference_command": subprocess.list2cmdline(sys.argv),
            "official_clip_weight_sha256": clip_metadata.weight_sha256,
            "class_mapping_sha256": mapping.digest(),
            "test_manifest_sha256": _sha256(test_manifest),
            "test_record_count": len(test_records),
            "inference_seconds": inference_seconds,
            "device": str(device),
            "inference_mode": "single_frozen_normalized_clip_head",
        },
        spec=SubmissionSpec(include_header=False),
    )
    print(
        json.dumps(
            {
                "csv_path": str(package.csv_path),
                "zip_path": str(package.zip_path),
                "manifest_path": str(package.manifest_path),
                "record_count": package.record_count,
                "csv_sha256": package.csv_sha256,
                "zip_sha256": package.zip_sha256,
                "inference_seconds": inference_seconds,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
