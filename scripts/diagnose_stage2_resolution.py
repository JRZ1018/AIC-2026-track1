"""Paired 224/336 inference-only screen on Stage 2 train-derived validation.

The frozen checkpoint was trained at 224. A negative result does not rule out
matched 336 training; this script only checks whether a cheap inference-only
resolution change is promising before launching another training experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.class_mapping import ClassMapping  # noqa: E402
from aictrack1.data.contracts import load_dataset_config  # noqa: E402
from aictrack1.data.indexed_images import read_indexed_images  # noqa: E402
from aictrack1.data.manifest import sha256_file  # noqa: E402
from aictrack1.data.transforms import build_split_transforms  # noqa: E402
from aictrack1.engine.training_queue import TrainingLease  # noqa: E402
from aictrack1.evaluation.classification import classification_diagnostics  # noqa: E402
from aictrack1.evaluation.long_tail import long_tail_diagnostics, stratify_classes  # noqa: E402
from aictrack1.models.clip_resolution import enable_interpolated_visual_resolution  # noqa: E402
from aictrack1.models.single_model import load_diagnostic_model, resolve_relative  # noqa: E402


class PairedResolutionDataset(Dataset):
    def __init__(self, records, transform_224, transform_336, allow_truncated_images):
        self.records = records
        self.transform_224 = transform_224
        self.transform_336 = transform_336
        self.allow_truncated_images = allow_truncated_images

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        if record.source != "train" or record.partition != "validation":
            raise ValueError("only Stage 2 train-derived validation is allowed")
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = self.allow_truncated_images
            with Image.open(record.absolute_path) as opened:
                image = opened.convert("RGB")
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous
        return {
            "image_224": self.transform_224(image),
            "image_336": self.transform_336(image),
            "target": int(record.internal_class_index),
            "sample_id": record.sample_id,
        }


def summarize(predictions, targets, stratification, num_classes):
    overall = classification_diagnostics(predictions, targets, num_classes=num_classes)
    tail = long_tail_diagnostics(
        predictions, targets, num_classes=num_classes, stratification=stratification
    )["strata"]["tail"]
    return {
        "top1": overall["top1"],
        "macro_accuracy": overall["macro_accuracy"],
        "zero_recall_classes": overall["zero_recall_classes"],
        "tail_class_mean_recall": tail["evaluated_class_mean_accuracy"],
        "tail_sample_accuracy": tail["accuracy"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None, help="smoke only")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or (args.max_samples is not None and args.max_samples < 1):
        raise ValueError("invalid batch size, workers or smoke sample count")
    output = args.output_dir.resolve()
    if not output.is_relative_to((ROOT / "artifacts" / "diagnostics").resolve()):
        raise ValueError("output must be under artifacts/diagnostics")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    config_path = args.experiment_config.resolve()
    experiment = yaml.safe_load(config_path.read_text(encoding="utf-8"))["experiment"]
    if experiment.get("competition_stage") != "stage2" or experiment.get("training_scope") != "fixed_split_validation":
        raise ValueError("requires a Stage 2 fixed-split model")
    checkpoint_path = args.checkpoint.resolve()
    if checkpoint_path.name != "best.pt" or sha256_file(checkpoint_path) != args.expected_checkpoint_sha256:
        raise ValueError("fixed-split checkpoint SHA-256 mismatch")
    run_output = resolve_relative(config_path.parent, experiment["output_dir"])
    if checkpoint_path.parent != run_output:
        raise ValueError("checkpoint must belong to experiment run")
    prior = json.loads((run_output / "metrics.json").read_text(encoding="utf-8"))
    if prior.get("selected_checkpoint_sha256") != args.expected_checkpoint_sha256:
        raise ValueError("metrics select a different checkpoint")
    dataset_config = load_dataset_config(resolve_relative(config_path.parent, experiment["dataset_config"]))
    if dataset_config.competition_stage != "stage2" or dataset_config.transforms.image_size != 224:
        raise ValueError("baseline must be Stage 2 at native 224 pixels")
    mapping = ClassMapping.from_csv(dataset_config.class_mapping)
    manifest_path = resolve_relative(config_path.parent, experiment["manifest"])
    split_path = resolve_relative(config_path.parent, experiment["split"])
    records = read_indexed_images(
        dataset_config, source="train", manifest_path=manifest_path, split_path=split_path
    )
    if len(records) != dataset_config.expected_train_count:
        raise ValueError("incomplete Stage 2 train manifest")
    counts = torch.bincount(
        torch.tensor([int(r.internal_class_index) for r in records if r.partition == "train"]),
        minlength=len(mapping),
    )
    validation = [r for r in records if r.partition == "validation"]
    full_validation_count = len(validation)
    if args.max_samples is not None:
        validation = validation[: args.max_samples]
    if not validation:
        raise ValueError("empty validation partition")
    if not torch.cuda.is_available():
        raise RuntimeError("local CUDA is unavailable")
    stratification = stratify_classes(
        counts, head_fraction=0.2, mid_fraction=0.3, seed=dataset_config.split.seed
    )
    lease = TrainingLease.acquire(
        ROOT / "artifacts" / "queues" / "training.lock.json",
        owner="stage2-336-inference-only-diagnostic",
        config=str(config_path),
        output_dir=str(output),
        command=tuple(sys.argv),
    )
    try:
        device = torch.device("cuda")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("training_scope") != "fixed_split_validation":
            raise ValueError("full-train checkpoint is forbidden")
        fitted = load_diagnostic_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            class_mapping_digest=mapping.digest(),
            num_classes=len(mapping),
            split_sha256=sha256_file(split_path),
            device=device,
            git_commit=str(checkpoint.get("git_commit") or "unknown_in_checkpoint"),
        )
        enable_interpolated_visual_resolution(fitted.model.visual, target_resolution=336)
        native_transform = build_split_transforms(dataset_config.transforms)["validation"]
        enlarged_config = replace(dataset_config.transforms, image_size=336, resize_size=336)
        enlarged_transform = build_split_transforms(enlarged_config)["validation"]
        loader = DataLoader(
            PairedResolutionDataset(
                validation, native_transform, enlarged_transform,
                dataset_config.allow_truncated_images,
            ),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
            pin_memory=True, persistent_workers=args.workers > 0,
        )
        predictions_224, predictions_336, targets, sample_ids = [], [], [], []
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for batch in loader:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits_224 = fitted.model(batch["image_224"].to(device, non_blocking=True))
                    logits_336 = fitted.model(batch["image_336"].to(device, non_blocking=True))
                predictions_224.append(logits_224.argmax(dim=1).cpu())
                predictions_336.append(logits_336.argmax(dim=1).cpu())
                targets.append(batch["target"].long())
                sample_ids.extend(batch["sample_id"])
        native = torch.cat(predictions_224)
        enlarged = torch.cat(predictions_336)
        target = torch.cat(targets)
        baseline = summarize(native, target, stratification, len(mapping))
        candidate = summarize(enlarged, target, stratification, len(mapping))
        if args.max_samples is None and abs(baseline["top1"] - float(prior["best_top1"])) > 0.0002:
            raise RuntimeError("224 baseline parity failed")
        delta = {
            key: (None if baseline[key] is None or candidate[key] is None else candidate[key] - baseline[key])
            for key in ("top1", "macro_accuracy", "tail_class_mean_recall", "tail_sample_accuracy")
        }
        center_correct, enlarged_correct = native.eq(target), enlarged.eq(target)
        passed = (
            args.max_samples is None
            and delta["top1"] >= 0.003
            and delta["macro_accuracy"] >= 0
            and delta["tail_class_mean_recall"] is not None
            and delta["tail_class_mean_recall"] >= 0
            and candidate["zero_recall_classes"] <= baseline["zero_recall_classes"]
        )
        report = {
            "scope": "Stage 2 train-derived fixed validation only; no test access",
            "protocol": "same frozen 224-trained checkpoint; native 224 versus 336 with 7x7-to-10x10 position interpolation",
            "smoke_only": args.max_samples is not None,
            "full_validation_count": full_validation_count,
            "evaluated_count": len(validation),
            "config_sha256": sha256_file(config_path),
            "checkpoint_sha256": fitted.checkpoint_sha256,
            "split_sha256": sha256_file(split_path),
            "manifest_sha256": sha256_file(manifest_path),
            "class_mapping_sha256": mapping.digest(),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_vram_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
            "native_224": baseline,
            "interpolated_336": candidate,
            "delta_336_minus_224": delta,
            "paired_wins": int((enlarged_correct & ~center_correct).sum()),
            "paired_losses": int((center_correct & ~enlarged_correct).sum()),
            "changed_predictions": int((native != enlarged).sum()),
            "inference_only_screen_passed": passed,
        }
        output.mkdir(parents=True, exist_ok=True)
        with (output / "paired_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("sample_id", "target", "prediction_224", "prediction_336"))
            writer.writerows(zip(sample_ids, target.tolist(), native.tolist(), enlarged.tolist(), strict=True))
        temporary = output / "report.json.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output / "report.json")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        lease.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
