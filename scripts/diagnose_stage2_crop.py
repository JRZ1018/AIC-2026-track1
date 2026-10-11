"""Paired center-crop/letterbox inference on Stage 2 train-derived validation only.

This diagnostic never accepts a test manifest or full-train checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.class_mapping import ClassMapping  # noqa: E402
from aictrack1.data.contracts import load_dataset_config  # noqa: E402
from aictrack1.data.indexed_images import IndexedImage, read_indexed_images  # noqa: E402
from aictrack1.data.manifest import sha256_file  # noqa: E402
from aictrack1.data.transforms import (  # noqa: E402
    CLIP_MEAN,
    _normalize,
    _safe_exif_transpose,
    build_split_transforms,
)
from aictrack1.evaluation.classification import classification_diagnostics  # noqa: E402
from aictrack1.evaluation.long_tail import stratify_classes  # noqa: E402
from aictrack1.models.single_model import load_diagnostic_model, resolve_relative  # noqa: E402


def letterbox(image: Image.Image, size: int) -> torch.Tensor:
    """Preserve the full RGB image inside a square CLIP-mean canvas."""
    image = _safe_exif_transpose(image).convert("RGB")
    width, height = image.size
    scale = size / max(width, height)
    resized_width = min(size, max(1, round(width * scale)))
    resized_height = min(size, max(1, round(height * scale)))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    fill = tuple(round(channel * 255) for channel in CLIP_MEAN)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(resized, ((size - resized_width) // 2, (size - resized_height) // 2))
    return _normalize(canvas)


class PairedValidationDataset(Dataset[dict[str, object]]):
    def __init__(self, records: list[IndexedImage], center_transform, image_size: int) -> None:
        self.records = records
        self.center_transform = center_transform
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        if record.partition != "validation" or record.source != "train":
            raise ValueError("diagnostic dataset must contain Stage 2 train-derived validation only")
        with Image.open(record.absolute_path) as opened:
            image = _safe_exif_transpose(opened).convert("RGB")
        width, height = image.size
        return {
            "center": self.center_transform(image),
            "letterbox": letterbox(image, self.image_size),
            "target": int(record.internal_class_index),
            "aspect_ratio": max(width, height) / min(width, height),
            "sample_id": record.sample_id,
        }


def _metrics(predictions: torch.Tensor, targets: torch.Tensor, tail_mask: torch.Tensor, num_classes: int) -> dict:
    overall = classification_diagnostics(predictions, targets, num_classes=num_classes)
    correct = predictions.eq(targets)
    class_total = torch.bincount(targets, minlength=num_classes)
    class_correct = torch.bincount(targets[correct], minlength=num_classes)
    tail_classes = tail_mask & (class_total > 0)
    tail_recall = class_correct[tail_classes].float() / class_total[tail_classes]
    tail_samples = tail_mask[targets]
    return {
        "top1": overall["top1"],
        "macro_accuracy": overall["macro_accuracy"],
        "zero_recall_classes": overall["zero_recall_classes"],
        "tail_sample_accuracy": float(correct[tail_samples].float().mean()),
        "tail_macro_recall": float(tail_recall.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None, help="small smoke only; no gate")
    args = parser.parse_args()
    if args.workers < 0 or args.batch_size < 1:
        raise ValueError("invalid DataLoader settings")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max-samples must be positive")
    config_path = args.experiment_config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    experiment = yaml.safe_load(config_path.read_text(encoding="utf-8"))["experiment"]
    if experiment.get("competition_stage") != "stage2" or experiment.get("training_scope") != "fixed_split_validation":
        raise ValueError("requires a Stage 2 fixed-split experiment")
    if checkpoint_path.name != "best.pt" or sha256_file(checkpoint_path) != args.expected_checkpoint_sha256:
        raise ValueError("expected fixed-split best.pt SHA-256 mismatch")
    dataset_config = load_dataset_config(resolve_relative(config_path.parent, experiment["dataset_config"]))
    if dataset_config.competition_stage != "stage2":
        raise ValueError("only Stage 2 train data are allowed")
    mapping = ClassMapping.from_csv(dataset_config.class_mapping)
    manifest_path = resolve_relative(config_path.parent, experiment["manifest"])
    split_path = resolve_relative(config_path.parent, experiment["split"])
    records = read_indexed_images(
        dataset_config, source="train", manifest_path=manifest_path, split_path=split_path
    )
    if len(records) != dataset_config.expected_train_count:
        raise ValueError("incomplete train manifest")
    train_counts = torch.bincount(
        torch.tensor([int(record.internal_class_index) for record in records if record.partition == "train"]),
        minlength=len(mapping),
    )
    validation = [record for record in records if record.partition == "validation"]
    full_count = len(validation)
    if args.max_samples is not None:
        validation = validation[: args.max_samples]
    if not validation:
        raise ValueError("empty validation partition")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required on the Shanghai L20")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("training_scope") != "fixed_split_validation":
        raise ValueError("full-train checkpoint is forbidden")
    model_info = load_diagnostic_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        class_mapping_digest=mapping.digest(),
        num_classes=len(mapping),
        split_sha256=sha256_file(split_path),
        device=device,
        git_commit=str(checkpoint.get("git_commit") or "unknown_in_checkpoint"),
    )
    transform = build_split_transforms(dataset_config.transforms)["validation"]
    loader = DataLoader(
        PairedValidationDataset(validation, transform, dataset_config.transforms.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    predictions: dict[str, list[torch.Tensor]] = {"center": [], "letterbox": []}
    targets: list[torch.Tensor] = []
    aspect_ratios: list[torch.Tensor] = []
    sample_ids: list[str] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            for key in ("center", "letterbox"):
                images = batch[key].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model_info.model(images)
                predictions[key].append(logits.argmax(dim=1).cpu())
            targets.append(batch["target"].long())
            aspect_ratios.append(batch["aspect_ratio"].float())
            sample_ids.extend(batch["sample_id"])
    predictions = {key: torch.cat(chunks) for key, chunks in predictions.items()}
    target = torch.cat(targets)
    aspect = torch.cat(aspect_ratios)
    strata = stratify_classes(train_counts, head_fraction=0.2, mid_fraction=0.3, seed=dataset_config.split.seed)
    tail_mask = torch.tensor([assignment == "tail" for assignment in strata.assignments])
    center_metrics = _metrics(predictions["center"], target, tail_mask, len(mapping))
    letterbox_metrics = _metrics(predictions["letterbox"], target, tail_mask, len(mapping))
    center_correct = predictions["center"].eq(target)
    letterbox_correct = predictions["letterbox"].eq(target)
    slices = {}
    for name, mask in (("aspect_le_1_5", aspect <= 1.5), ("aspect_gt_1_5", aspect > 1.5)):
        slices[name] = {
            "count": int(mask.sum()),
            "center_top1": float(center_correct[mask].float().mean()),
            "letterbox_top1": float(letterbox_correct[mask].float().mean()),
            "letterbox_wins": int((letterbox_correct[mask] & ~center_correct[mask]).sum()),
            "center_wins": int((center_correct[mask] & ~letterbox_correct[mask]).sum()),
        }
    delta_top1 = letterbox_metrics["top1"] - center_metrics["top1"]
    delta_macro = letterbox_metrics["macro_accuracy"] - center_metrics["macro_accuracy"]
    delta_tail = letterbox_metrics["tail_macro_recall"] - center_metrics["tail_macro_recall"]
    gate = (
        args.max_samples is None
        and delta_top1 >= 0.003
        and delta_macro >= 0.003
        and delta_tail >= -0.002
        and letterbox_metrics["zero_recall_classes"] <= center_metrics["zero_recall_classes"]
        and slices["aspect_gt_1_5"]["letterbox_wins"] > slices["aspect_gt_1_5"]["center_wins"]
    )
    report = {
        "scope": "Stage 2 train-derived fixed validation only; no test access",
        "full_validation_count": full_count,
        "evaluated_count": len(validation),
        "smoke_only": args.max_samples is not None,
        "experiment_config_sha256": sha256_file(config_path),
        "checkpoint_sha256": model_info.checkpoint_sha256,
        "split_sha256": sha256_file(split_path),
        "manifest_sha256": sha256_file(manifest_path),
        "class_mapping_sha256": mapping.digest(),
        "elapsed_seconds": time.perf_counter() - started,
        "center": center_metrics,
        "letterbox": letterbox_metrics,
        "slices": slices,
        "delta_letterbox_minus_center": {
            "top1": delta_top1,
            "macro_accuracy": delta_macro,
            "tail_macro_recall": delta_tail,
        },
        "pre_registered_gate_passed": gate,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "paired_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("sample_id", "target", "aspect_ratio", "center_prediction", "letterbox_prediction"))
        writer.writerows(
            zip(sample_ids, target.tolist(), aspect.tolist(), predictions["center"].tolist(), predictions["letterbox"].tolist(), strict=True)
        )
    temp_report = output_dir / "report.json.tmp"
    temp_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_report, output_dir / "report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
