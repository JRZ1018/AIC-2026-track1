"""Train one reproducible single-model PEFT candidate from the A1 cosine head."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.class_mapping import ClassMapping
from aictrack1.data.contracts import ensure_output_path_is_safe, load_dataset_config
from aictrack1.data.indexed_images import IndexedImage, IndexedImageDataset, read_indexed_images
from aictrack1.data.transforms import build_split_transforms
from aictrack1.engine.peft import load_trainable_state_dict, trainable_state_dict
from aictrack1.evaluation.classification import calibration_diagnostics, classification_diagnostics
from aictrack1.models.clip_vit_b32 import load_clip_vit_b32
from aictrack1.models.heads import CosineClassifier
from aictrack1.models.peft import PEFTImageClassifier, configure_visual_peft


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_atomic(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _model_settings(path: Path) -> tuple[Path, str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = raw.get("model", raw)
    if raw.get("provider") != "OpenAI" or raw.get("name") != "ViT-B/32":
        raise ValueError("model config must declare OpenAI ViT-B/32")
    if raw.get("runtime_downloads_allowed") is not False:
        raise ValueError("runtime downloads must be disabled")
    return _resolve(path.parent, str(raw["weight_path"])), str(raw["expected_weight_sha256"])


def _conflict_ids(manifest: Path, conflicts: Path) -> set[str]:
    with conflicts.open("r", encoding="utf-8-sig", newline="") as handle:
        hashes = {str(row["sha256"]) for row in csv.DictReader(handle)}
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            str(row["sample_id"])
            for row in csv.DictReader(handle)
            if str(row["sha256"]) in hashes
        }


def _loader(
    dataset: IndexedImageDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


@torch.inference_mode()
def _evaluate(
    model: PEFTImageClassifier,
    records: list[IndexedImage],
    dataset: IndexedImageDataset,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
    num_classes: int,
    conflict_ids: set[str],
) -> dict[str, Any]:
    model.eval()
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in _loader(
        dataset,
        batch_size=batch_size,
        workers=workers,
        shuffle=False,
        seed=0,
        device=device,
    ):
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits.append(model(images).float().cpu())
        targets.append(batch["label"])
    all_logits = torch.cat(logits)
    all_targets = torch.cat(targets).long()
    predictions = all_logits.argmax(dim=1)
    conflict_mask = torch.tensor([record.sample_id in conflict_ids for record in records])
    return {
        "all": classification_diagnostics(predictions, all_targets, num_classes=num_classes),
        "calibration": calibration_diagnostics(all_logits, all_targets),
        "cross_label_exact_conflicts": classification_diagnostics(
            predictions[conflict_mask], all_targets[conflict_mask], num_classes=num_classes
        ),
        "remaining": classification_diagnostics(
            predictions[~conflict_mask], all_targets[~conflict_mask], num_classes=num_classes
        ),
        "conflict_sample_count": int(conflict_mask.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    raw_document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw = raw_document.get("experiment", raw_document)
    base = config_path.parent
    dataset_path = _resolve(base, str(raw["dataset_config"]))
    model_path = _resolve(base, str(raw["model_config"]))
    manifest_path = _resolve(base, str(raw["manifest"]))
    split_path = _resolve(base, str(raw["split"]))
    conflicts_path = _resolve(base, str(raw["label_conflicts"]))
    a1_checkpoint_path = _resolve(base, str(raw["a1_head_checkpoint"]))
    output = _resolve(base, str(raw["output_dir"]))
    dataset_config = load_dataset_config(dataset_path)
    ensure_output_path_is_safe(dataset_config, output)
    if output.exists() and args.resume is None:
        raise FileExistsError(f"PEFT output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    training = raw["training"]
    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    workers = int(training.get("workers", 0))
    seed = int(training["seed"])
    if epochs < 1 or batch_size < 1 or workers < 0:
        raise ValueError("invalid epoch, batch-size, or worker configuration")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(str(raw.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    records = read_indexed_images(
        dataset_config,
        source="train",
        manifest_path=manifest_path,
        split_path=split_path,
    )
    train_records = [record for record in records if record.partition == "train"]
    validation_records = [record for record in records if record.partition == "validation"]
    transforms = build_split_transforms(dataset_config.transforms)
    train_dataset = IndexedImageDataset(
        train_records,
        transforms["train"],
        allow_truncated_images=dataset_config.allow_truncated_images,
        stochastic_seed=seed,
    )
    validation_dataset = IndexedImageDataset(
        validation_records,
        transforms["validation"],
        allow_truncated_images=dataset_config.allow_truncated_images,
    )
    mapping = ClassMapping.from_csv(dataset_config.class_mapping)
    num_classes = len(mapping)
    weight_path, expected_weight_sha256 = _model_settings(model_path)
    clip_model, _, clip_metadata = load_clip_vit_b32(weight_path, expected_weight_sha256, device)
    visual = clip_model.visual.float()
    del clip_model
    peft = raw["peft"]
    summary = configure_visual_peft(
        visual,
        str(peft["mode"]),
        lora_rank=int(peft.get("lora_rank", 4)),
        lora_alpha=float(peft.get("lora_alpha", 8.0)),
    )
    head = CosineClassifier(clip_metadata.feature_dim, num_classes)
    a1 = torch.load(a1_checkpoint_path, map_location="cpu", weights_only=False)
    if a1["head"]["kind"] != "cosine":
        raise ValueError("A1 initializer is not a cosine head")
    metadata = a1["feature_cache_metadata"]
    if metadata["clip_weight_sha256"] != clip_metadata.weight_sha256:
        raise ValueError("A1 head and PEFT backbone weight hashes differ")
    if metadata["class_mapping_sha256"] != mapping.digest():
        raise ValueError("A1 head and current class mapping differ")
    if metadata["split_sha256"] != _sha256(split_path):
        raise ValueError("A1 head and current split differ")
    head.load_state_dict(a1["model"])
    model = PEFTImageClassifier(visual, head).to(device)
    visual_parameters = [parameter for parameter in visual.parameters() if parameter.requires_grad]
    head_parameters = list(head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": visual_parameters, "lr": float(training["visual_learning_rate"])},
            {"params": head_parameters, "lr": float(training["head_learning_rate"])},
        ],
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    conflict_ids = _conflict_ids(manifest_path, conflicts_path)
    config_sha256 = _sha256(config_path)
    epoch = 0
    history: list[dict[str, Any]] = []
    best_top1 = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None

    if args.resume is not None:
        state = torch.load(args.resume.resolve(), map_location="cpu", weights_only=False)
        if state["config_sha256"] != config_sha256:
            raise ValueError("PEFT resume configuration hash mismatch")
        load_trainable_state_dict(model, state["trainable_state"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        epoch = int(state["epoch"])
        history = state["history"]
        best_top1 = float(state["best_top1"])
        best_state = state["best_state"]
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.random.set_rng_state(state["torch_rng_state"])
        if "cuda_rng_state_all" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])

    if epoch == 0:
        initial = _evaluate(
            model,
            validation_records,
            validation_dataset,
            batch_size=batch_size,
            workers=workers,
            device=device,
            num_classes=num_classes,
            conflict_ids=conflict_ids,
        )
        expected = float(raw["expected_initial_top1"])
        tolerance = float(raw.get("initial_top1_tolerance", 0.002))
        if abs(float(initial["all"]["top1"]) - expected) > tolerance:
            raise RuntimeError(
                f"raw-image A1 parity failed: got {initial['all']['top1']}, expected {expected}±{tolerance}"
            )
    else:
        initial = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_started = time.perf_counter()
    while epoch < epochs:
        train_dataset.set_epoch(epoch)
        model.train()
        loader = _loader(
            train_dataset,
            batch_size=batch_size,
            workers=workers,
            shuffle=True,
            seed=seed + epoch,
            device=device,
        )
        train_started = time.perf_counter()
        loss_sum = 0.0
        count = 0
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(images), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("PEFT training produced a non-finite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [*visual_parameters, *head_parameters],
                float(training["gradient_clip_norm"]),
            )
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(labels)
            count += len(labels)
        train_seconds = time.perf_counter() - train_started
        scheduler.step()
        epoch += 1
        validation = _evaluate(
            model,
            validation_records,
            validation_dataset,
            batch_size=batch_size,
            workers=workers,
            device=device,
            num_classes=num_classes,
            conflict_ids=conflict_ids,
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / count,
            "train_seconds": train_seconds,
            "train_samples_per_second": count / train_seconds,
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            "validation": validation,
        }
        history.append(row)
        top1 = float(validation["all"]["top1"])
        if top1 > best_top1:
            best_top1 = top1
            best_state = copy.deepcopy(trainable_state_dict(model))
        checkpoint: dict[str, Any] = {
            "config_sha256": config_sha256,
            "epoch": epoch,
            "trainable_state": trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
            "best_top1": best_top1,
            "best_state": best_state,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.random.get_rng_state().clone(),
        }
        if torch.cuda.is_available():
            checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        _save_atomic(checkpoint, output / "last.pt")

    if best_state is None:
        raise RuntimeError("PEFT training produced no best state")
    _save_atomic(
        {
            "config_sha256": config_sha256,
            "mode": peft["mode"],
            "clip_weight_sha256": clip_metadata.weight_sha256,
            "class_mapping_sha256": mapping.digest(),
            "split_sha256": _sha256(split_path),
            "trainable_state": best_state,
            "best_top1": best_top1,
        },
        output / "best.pt",
    )
    total_seconds = time.perf_counter() - total_started
    metrics = {
        "scope": "stage1 train/validation PEFT short screening; test data not accessed",
        "run_id": raw["run_id"],
        "peft": peft,
        "initial_validation": initial,
        "history": history,
        "best_top1": best_top1,
        "duration_seconds": total_seconds,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0,
        "train_count": len(train_records),
        "validation_count": len(validation_records),
        "total_visual_parameters": summary.total_visual_parameters,
        "trainable_visual_parameters": summary.trainable_visual_parameters,
        "trainable_total_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "best_checkpoint_sha256": _sha256(output / "best.pt"),
        "last_checkpoint_sha256": _sha256(output / "last.pt"),
        "config_sha256": config_sha256,
        "dataset_config_sha256": _sha256(dataset_path),
        "manifest_sha256": _sha256(manifest_path),
        "split_sha256": _sha256(split_path),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copy2(config_path, output / "config.yaml")
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
