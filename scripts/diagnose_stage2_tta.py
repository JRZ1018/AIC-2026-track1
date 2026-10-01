"""Paired single-checkpoint flip TTA on Stage 2 train-derived validation only.

This diagnostic deliberately has no test-manifest or full-train checkpoint path.
It acquires the repository's atomic training lock while using the GPU.
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.class_mapping import ClassMapping  # noqa: E402
from aictrack1.data.contracts import load_dataset_config  # noqa: E402
from aictrack1.data.indexed_images import IndexedImageDataset, read_indexed_images  # noqa: E402
from aictrack1.data.manifest import sha256_file  # noqa: E402
from aictrack1.data.transforms import build_split_transforms  # noqa: E402
from aictrack1.engine.training_queue import TrainingLease  # noqa: E402
from aictrack1.evaluation.classification import classification_diagnostics  # noqa: E402
from aictrack1.evaluation.four_view import (  # noqa: E402
    PairedViewDataset,
    fixed_four_view_logits,
    resolve_four_view_resize,
)
from aictrack1.evaluation.long_tail import long_tail_diagnostics, stratify_classes  # noqa: E402
from aictrack1.models.single_model import load_diagnostic_model, resolve_relative  # noqa: E402


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


class PairedScaleValidationDataset(Dataset):
    """Two deterministic 224 crops from each Stage2 train-derived validation image."""

    def __init__(self, records, native_transform, scaled_transform):
        self.records = records
        self.native_transform = native_transform
        self.scaled_transform = scaled_transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        if record.source != "train" or record.partition != "validation":
            raise ValueError("paired-scale diagnostic requires train-derived validation")
        with Image.open(record.absolute_path) as opened:
            return {
                "image": self.native_transform(opened),
                "scaled_image": self.scaled_transform(opened),
                "label": int(record.internal_class_index),
                "sample_id": record.sample_id,
            }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument(
        "--raw-ema-diagnostic", action="store_true",
        help="Allow this run's raw_best.pt only for a paired EMA-versus-raw train-validation audit",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--hard-seconds", type=int, default=None,
                        help="Abort at a batch boundary before this wall-clock limit; releases the diagnostic lock")
    parser.add_argument("--max-samples", type=int, default=None, help="smoke only; cannot pass gate")
    parser.add_argument(
        "--save-fused-logits",
        action="store_true",
        help="atomically save the selected fused validation logits for train-only audits",
    )
    four_view = parser.add_mutually_exclusive_group()
    four_view.add_argument(
        "--four-view-scale256", action="store_true", help="C7 fixed two-scale four-view validation"
    )
    four_view.add_argument(
        "--four-view-scale384",
        action="store_true",
        help="C8 fixed two-scale four-view validation on a 336-trained checkpoint",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or (args.max_samples is not None and args.max_samples < 1):
        raise ValueError("invalid batch size, workers or smoke sample count")
    if args.hard_seconds is not None and args.hard_seconds < 1:
        raise ValueError("hard-seconds must be positive")
    deadline = time.monotonic() + args.hard_seconds if args.hard_seconds is not None else None
    output = args.output_dir.resolve()
    if not output.is_relative_to((ROOT / "artifacts" / "diagnostics").resolve()):
        raise ValueError("diagnostic output must be under artifacts/diagnostics")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    config_path = args.experiment_config.resolve()
    experiment = yaml.safe_load(config_path.read_text(encoding="utf-8"))["experiment"]
    if experiment.get("competition_stage") != "stage2" or experiment.get("training_scope") != "fixed_split_validation":
        raise ValueError("requires a Stage 2 fixed-split validation experiment")
    checkpoint_path = args.checkpoint.resolve()
    expected_name = "raw_best.pt" if args.raw_ema_diagnostic else "best.pt"
    if checkpoint_path.name != expected_name or sha256_file(checkpoint_path) != args.expected_checkpoint_sha256:
        raise ValueError(f"expected fold-specific {expected_name} SHA-256 mismatch")
    run_output = resolve_relative(config_path.parent, experiment["output_dir"])
    if checkpoint_path.parent != run_output:
        raise ValueError("checkpoint must belong to the experiment run")
    prior = json.loads((run_output / "metrics.json").read_text(encoding="utf-8"))
    if prior.get("selected_checkpoint") != "best.pt":
        raise ValueError("run metrics do not select best.pt")
    if args.raw_ema_diagnostic:
        if not bool((experiment.get("trainable_ema") or {}).get("enabled", False)):
            raise ValueError("raw_best.pt diagnostic requires an EMA run")
    elif prior.get("selected_checkpoint_sha256") != args.expected_checkpoint_sha256:
        raise ValueError("run metrics do not select this checkpoint")
    dataset_config = load_dataset_config(resolve_relative(config_path.parent, experiment["dataset_config"]))
    if dataset_config.competition_stage != "stage2":
        raise ValueError("Stage 2 official train data required")
    visual_resolution = int(experiment.get("visual_input_resolution", 224))
    if visual_resolution != dataset_config.transforms.image_size:
        raise ValueError("TTA visual resolution must match validation transform")
    four_view_option = (
        "scale256"
        if args.four_view_scale256
        else "scale384"
        if args.four_view_scale384
        else None
    )
    four_view_resize = resolve_four_view_resize(visual_resolution, four_view_option)
    mapping = ClassMapping.from_csv(dataset_config.class_mapping)
    manifest_path = resolve_relative(config_path.parent, experiment["manifest"])
    split_path = resolve_relative(config_path.parent, experiment["split"])
    records = read_indexed_images(
        dataset_config, source="train", manifest_path=manifest_path, split_path=split_path
    )
    if len(records) != dataset_config.expected_train_count:
        raise ValueError("incomplete Stage 2 train manifest")
    train_counts = torch.bincount(
        torch.tensor([int(r.internal_class_index) for r in records if r.partition == "train"]),
        minlength=len(mapping),
    )
    validation = [r for r in records if r.partition == "validation"]
    full_validation_count = len(validation)
    if args.max_samples is not None:
        validation = validation[: args.max_samples]
    if not validation or any(r.source != "train" for r in validation):
        raise ValueError("requires nonempty Stage 2 train-derived validation")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("diagnostic wall-clock limit reached during preflight")
    device = torch.device("cuda")
    stratification = stratify_classes(
        train_counts, head_fraction=0.2, mid_fraction=0.3, seed=dataset_config.split.seed
    )
    lease = TrainingLease.acquire(
        ROOT / "artifacts" / "queues" / "training.lock.json",
        owner="stage2-single-checkpoint-tta-diagnostic",
        config=str(config_path),
        output_dir=str(output),
        command=tuple(sys.argv),
    )
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("training_scope") != "fixed_split_validation":
            raise ValueError("full-train checkpoint is forbidden")
        if args.raw_ema_diagnostic and (
            checkpoint.get("selection_rule") != "raw best fixed-split validation Top-1; diagnostic only"
            or checkpoint.get("trainable_ema") is not None
        ):
            raise ValueError("raw_best.pt is not the raw diagnostic state of this EMA run")
        fitted = load_diagnostic_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            class_mapping_digest=mapping.digest(),
            num_classes=len(mapping),
            split_sha256=sha256_file(split_path),
            device=device,
            git_commit=str(checkpoint.get("git_commit") or "unknown_in_checkpoint"),
        )
        native_transform = build_split_transforms(dataset_config.transforms)["validation"]
        if four_view_resize is not None:
            scaled_transform = build_split_transforms(
                replace(dataset_config.transforms, resize_size=four_view_resize)
            )["validation"]
            if four_view_option == "scale256":
                # Frozen C7/224 candidate: keep the historical loader untouched so
                # its published fold reports stay reproducible.
                dataset = PairedScaleValidationDataset(
                    validation, native_transform, scaled_transform
                )
            else:
                dataset = PairedViewDataset(
                    validation,
                    native_transform,
                    scaled_transform,
                    required_partition="validation",
                    allow_truncated_images=dataset_config.allow_truncated_images,
                )
        else:
            dataset = IndexedImageDataset(
                validation, native_transform,
                allow_truncated_images=dataset_config.allow_truncated_images,
            )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
            pin_memory=True, persistent_workers=args.workers > 0,
        )
        center_predictions, tta_predictions, targets, sample_ids = [], [], [], []
        four_view_predictions = []
        fused_logits_chunks = []
        started = time.perf_counter()
        with torch.inference_mode():
            for batch in loader:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("diagnostic wall-clock limit reached; no partial report is valid")
                images = batch["image"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    if four_view_resize is not None:
                        scaled = batch["scaled_image"].to(device, non_blocking=True)
                        center_logits, two_logits, four_logits = fixed_four_view_logits(
                            fitted.model, images, scaled
                        )
                    else:
                        center_logits = fitted.model(images).float()
                        flipped_logits = fitted.model(torch.flip(images, dims=(-1,))).float()
                        two_logits = (center_logits + flipped_logits) / 2
                center_predictions.append(center_logits.argmax(dim=1).cpu())
                tta_predictions.append(two_logits.argmax(dim=1).cpu())
                if four_view_resize is not None:
                    four_view_predictions.append(four_logits.argmax(dim=1).cpu())
                if args.save_fused_logits:
                    fused = four_logits if four_view_resize is not None else two_logits
                    fused_logits_chunks.append(fused.detach().to(device="cpu", dtype=torch.float16))
                targets.append(batch["label"].long())
                sample_ids.extend(batch["sample_id"])
        center = torch.cat(center_predictions)
        tta = torch.cat(tta_predictions)
        target = torch.cat(targets)
        baseline = summarize(center, target, stratification, len(mapping))
        candidate = summarize(tta, target, stratification, len(mapping))
        four = torch.cat(four_view_predictions) if four_view_resize is not None else None
        four_metrics = summarize(four, target, stratification, len(mapping)) if four is not None else None
        expected_center_top1 = (
            float(checkpoint["best_top1"])
            if args.raw_ema_diagnostic else float(prior["best_top1"])
        )
        if args.max_samples is None and abs(baseline["top1"] - expected_center_top1) > 0.0002:
            raise RuntimeError("baseline parity with frozen validation metrics failed")
        delta = {
            key: (None if baseline[key] is None or candidate[key] is None else candidate[key] - baseline[key])
            for key in ("top1", "macro_accuracy", "tail_class_mean_recall", "tail_sample_accuracy")
        }
        if args.max_samples is None and delta["tail_class_mean_recall"] is None:
            raise RuntimeError("full validation unexpectedly has no tail-class recall")
        center_correct, tta_correct = center.eq(target), tta.eq(target)
        passed = (
            args.max_samples is None
            and delta["top1"] >= 0.003
            and delta["macro_accuracy"] >= 0.0
            and delta["tail_class_mean_recall"] >= 0.0
            and candidate["zero_recall_classes"] <= baseline["zero_recall_classes"]
        )
        report = {
            "scope": "Stage 2 train-derived fixed validation only; no test access",
            "view_protocol": f"native {visual_resolution} center view plus exact horizontal tensor flip; equal mean logits; one frozen checkpoint",
            "visual_input_resolution": visual_resolution,
            "smoke_only": args.max_samples is not None,
            "full_validation_count": full_validation_count,
            "evaluated_count": len(validation),
            "experiment_config_sha256": sha256_file(config_path),
            "checkpoint_sha256": fitted.checkpoint_sha256,
            "checkpoint_role": "raw_ema_diagnostic" if args.raw_ema_diagnostic else "selected_best",
            "split_sha256": sha256_file(split_path),
            "manifest_sha256": sha256_file(manifest_path),
            "class_mapping_sha256": mapping.digest(),
            "elapsed_seconds": time.perf_counter() - started,
            "center": baseline,
            "flip_tta": candidate,
            "delta_tta_minus_center": delta,
            "paired_wins": int((tta_correct & ~center_correct).sum()),
            "paired_losses": int((center_correct & ~tta_correct).sum()),
            "changed_predictions": int((tta != center).sum()),
            "pre_registered_first_fold_gate_passed": passed,
        }
        if four_metrics is not None:
            report["four_view_protocol"] = (
                f"native{visual_resolution} center and exact flip plus "
                f"resize_short_side{four_view_resize}_center_crop{visual_resolution} "
                "and exact flip; one checkpoint; four float logits equally averaged"
            )
            report["four_view_resize"] = four_view_resize
            report["four_view"] = four_metrics
            report["delta_four_minus_flip_tta"] = {
                key: (
                    None if four_metrics[key] is None or candidate[key] is None
                    else four_metrics[key] - candidate[key]
                )
                for key in ("top1", "macro_accuracy", "tail_class_mean_recall", "tail_sample_accuracy")
            }
            report["four_view_changed_predictions"] = int((four != tta).sum())
            report["four_view_paired_wins"] = int((four.eq(target) & ~tta.eq(target)).sum())
            report["four_view_paired_losses"] = int((tta.eq(target) & ~four.eq(target)).sum())
        output.mkdir(parents=True, exist_ok=True)
        if args.save_fused_logits:
            fused_logits = torch.cat(fused_logits_chunks)
            if fused_logits.shape != (len(validation), len(mapping)):
                raise RuntimeError("fused logits have an unexpected shape")
            logits_path = output / "fused_logits.pt"
            logits_temporary = output / "fused_logits.pt.tmp"
            torch.save(
                {
                    "schema": "aictrack1.stage2_train_derived_fused_logits.v1",
                    "scope": "Stage 2 train-derived fixed validation only; no test access",
                    "logits": fused_logits,
                    "targets": target,
                    "train_class_counts": train_counts,
                    "checkpoint_sha256": fitted.checkpoint_sha256,
                    "experiment_config_sha256": sha256_file(config_path),
                    "split_sha256": sha256_file(split_path),
                    "manifest_sha256": sha256_file(manifest_path),
                    "class_mapping_sha256": mapping.digest(),
                    "view_protocol": (
                        report.get("four_view_protocol") or report["view_protocol"]
                    ),
                },
                logits_temporary,
            )
            os.replace(logits_temporary, logits_path)
            # Safe-load immediately so an interrupted or incompatible write cannot
            # silently become evidence for a later inference decision.
            verified_logits = torch.load(logits_path, map_location="cpu", weights_only=True)
            if verified_logits.get("schema") != "aictrack1.stage2_train_derived_fused_logits.v1":
                raise RuntimeError("saved fused logits failed schema verification")
            if tuple(verified_logits["logits"].shape) != tuple(fused_logits.shape):
                raise RuntimeError("saved fused logits failed shape verification")
            report["fused_logits"] = {
                "path": str(logits_path),
                "sha256": sha256_file(logits_path),
                "dtype": str(fused_logits.dtype),
                "shape": list(fused_logits.shape),
                "weights_only_safe_load_verified": True,
            }
        with (output / "paired_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            if four is None:
                writer.writerow(("sample_id", "target", "center_prediction", "flip_tta_prediction"))
                writer.writerows(zip(sample_ids, target.tolist(), center.tolist(), tta.tolist(), strict=True))
            else:
                writer.writerow(("sample_id", "target", "center_prediction", "flip_tta_prediction", "four_view_prediction"))
                writer.writerows(zip(sample_ids, target.tolist(), center.tolist(), tta.tolist(), four.tolist(), strict=True))
        temporary = output / "report.json.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output / "report.json")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        lease.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
