"""Run one final test inference and atomically build the strict submission package."""

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
from aictrack1.data.transforms import build_split_transforms
from aictrack1.engine.peft import load_trainable_state_dict
from aictrack1.models.clip_vit_b32 import load_clip_vit_b32
from aictrack1.models.heads import load_classifier_head_checkpoint
from aictrack1.models.peft import (
    PEFTImageClassifier,
    configure_visual_peft,
    fold_visual_lora,
)
from aictrack1.submission import SubmissionSpec, build_submission_package


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256_variants(path: Path) -> set[str]:
    """Return hashes differing only by platform text line endings."""

    raw = path.read_bytes()
    text = raw.decode("utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return {
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        hashlib.sha256(normalized.replace("\n", "\r\n").encode("utf-8")).hexdigest(),
    }


def _model_settings(path: Path) -> tuple[Path, str]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = document.get("model", document)
    if raw.get("provider") != "OpenAI" or raw.get("name") != "ViT-B/32":
        raise ValueError("model config must declare OpenAI ViT-B/32")
    if raw.get("runtime_downloads_allowed") is not False:
        raise ValueError("runtime downloads must be disabled")
    return _resolve(path.parent, str(raw["weight_path"])), str(raw["expected_weight_sha256"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--test-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")

    config_path = args.experiment_config.resolve()
    config_document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = config_document.get("experiment", config_document)
    base = config_path.parent
    dataset_path = _resolve(base, str(experiment["dataset_config"]))
    model_path = _resolve(base, str(experiment["model_config"]))
    head_checkpoint_setting = experiment.get(
        "head_checkpoint", experiment.get("a1_head_checkpoint")
    )
    if head_checkpoint_setting is None:
        raise ValueError("experiment must define head_checkpoint or legacy a1_head_checkpoint")
    if "head_checkpoint" in experiment and "a1_head_checkpoint" in experiment:
        raise ValueError("experiment must not define both head checkpoint fields")
    head_path = _resolve(base, str(head_checkpoint_setting))
    split_path = _resolve(base, str(experiment["split"]))
    checkpoint_path = args.checkpoint.resolve()
    test_manifest = args.test_manifest.resolve()
    dataset_config = load_dataset_config(dataset_path)
    output = ensure_output_path_is_safe(dataset_config, args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"submission output directory is not empty: {output}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_sha256 = _sha256(config_path)
    if checkpoint.get("config_sha256") not in _text_sha256_variants(config_path):
        raise ValueError(
            "checkpoint and experiment configuration hashes differ beyond LF/CRLF normalization"
        )
    peft = experiment["peft"]
    if checkpoint.get("mode") != peft.get("mode"):
        raise ValueError("checkpoint and experiment PEFT modes differ")
    if peft.get("mode") != "lora_last_attention":
        raise ValueError("final inference currently requires the selected LoRA PEFT mode")

    mapping = ClassMapping.from_csv(dataset_config.class_mapping)
    if checkpoint.get("class_mapping_sha256") != mapping.digest():
        raise ValueError("checkpoint and class mapping hashes differ")
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

    device = torch.device(str(experiment.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    weight_path, expected_weight_sha256 = _model_settings(model_path)
    clip_model, _, clip_metadata = load_clip_vit_b32(weight_path, expected_weight_sha256, device)
    if checkpoint.get("clip_weight_sha256") != clip_metadata.weight_sha256:
        raise ValueError("checkpoint and official CLIP weight hashes differ")
    visual = clip_model.visual.float()
    del clip_model
    configure_visual_peft(
        visual,
        str(peft["mode"]),
        lora_rank=int(peft.get("lora_rank", 4)),
        lora_alpha=float(peft.get("lora_alpha", 8.0)),
    )
    head, head_checkpoint = load_classifier_head_checkpoint(
        head_path,
        feature_dim=clip_metadata.feature_dim,
        num_classes=len(mapping),
        expected_clip_weight_sha256=clip_metadata.weight_sha256,
        expected_class_mapping_sha256=mapping.digest(),
        expected_split_sha256=_sha256(split_path),
    )
    expected_head_sha256 = checkpoint.get("head_initializer_sha256")
    if expected_head_sha256 is not None and expected_head_sha256 != _sha256(head_path):
        raise ValueError("checkpoint and classifier head initializer hashes differ")
    normalized_features = head_checkpoint["feature_cache_metadata"].get("normalized")
    if not isinstance(normalized_features, bool):
        raise ValueError("head checkpoint must declare whether cached features were normalized")
    checkpoint_normalized = checkpoint.get("normalize_features")
    if checkpoint_normalized is not None and checkpoint_normalized != normalized_features:
        raise ValueError("checkpoint and classifier head feature normalization differ")
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    model = PEFTImageClassifier(
        visual,
        head,
        normalize_features=normalized_features,
    ).to(device)
    load_trainable_state_dict(model, checkpoint["trainable_state"])
    fold_visual_lora(model.visual)
    model.eval()

    transforms = build_split_transforms(dataset_config.transforms)
    test_dataset = IndexedImageDataset(
        test_records,
        transforms["validation"],
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
    predictions = dict(zip(filenames, predicted_indices, strict=True))
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    command = subprocess.list2cmdline(sys.argv)
    package = build_submission_package(
        predictions,
        filenames,
        mapping,
        output,
        run_metadata={
            "run_id": str(experiment["run_id"]),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "config_hash": config_sha256,
            "data_hash": _sha256(test_manifest),
            "git_commit": git_commit,
            "inference_command": command,
            "official_clip_weight_sha256": clip_metadata.weight_sha256,
            "class_mapping_sha256": mapping.digest(),
            "test_manifest_sha256": _sha256(test_manifest),
            "test_record_count": len(test_records),
            "inference_seconds": inference_seconds,
            "device": str(device),
            "lora_folded_for_single_model_inference": True,
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
