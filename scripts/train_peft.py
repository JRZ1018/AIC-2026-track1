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
from aictrack1.data.feature_cache import load_feature_cache
from aictrack1.data.indexed_images import IndexedImage, IndexedImageDataset, read_indexed_images
from aictrack1.data.transforms import build_split_transforms
from aictrack1.engine.peft import load_trainable_state_dict, trainable_state_dict
from aictrack1.evaluation.classification import calibration_diagnostics, classification_diagnostics
from aictrack1.methods.confidence import (
    ClasswiseHistoryWeighting,
    visual_prototype_soft_weights,
    weighted_cross_entropy,
)
from aictrack1.models.clip_vit_b32 import load_clip_vit_b32
from aictrack1.models.heads import load_classifier_head_checkpoint
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
    if raw.get("execution_enabled", True) is not True:
        raise RuntimeError("this PEFT configuration is design-only and not approved for execution")
    base = config_path.parent
    dataset_path = _resolve(base, str(raw["dataset_config"]))
    model_path = _resolve(base, str(raw["model_config"]))
    manifest_path = _resolve(base, str(raw["manifest"]))
    split_path = _resolve(base, str(raw["split"]))
    conflicts_path = _resolve(base, str(raw["label_conflicts"]))
    head_checkpoint_setting = raw.get("head_checkpoint", raw.get("a1_head_checkpoint"))
    if head_checkpoint_setting is None:
        raise ValueError("PEFT config must define head_checkpoint or legacy a1_head_checkpoint")
    if "head_checkpoint" in raw and "a1_head_checkpoint" in raw:
        raise ValueError("PEFT config must not define both head checkpoint fields")
    head_checkpoint_path = _resolve(base, str(head_checkpoint_setting))
    output = _resolve(base, str(raw["output_dir"]))
    dataset_config = load_dataset_config(dataset_path)
    ensure_output_path_is_safe(dataset_config, output)
    if output.exists() and args.resume is None:
        raise FileExistsError(f"PEFT output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    training = raw["training"]
    training_scope = str(raw.get("training_scope", "fixed_split_validation"))
    if training_scope not in {"fixed_split_validation", "full_train_fixed_epochs"}:
        raise ValueError(f"unsupported PEFT training scope {training_scope!r}")
    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    workers = int(training.get("workers", 0))
    seed = int(training["seed"])
    label_smoothing = float(training.get("label_smoothing", 0.0))
    if epochs < 1 or batch_size < 1 or workers < 0:
        raise ValueError("invalid epoch, batch-size, or worker configuration")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
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
    if training_scope == "full_train_fixed_epochs":
        expected_train_count = int(raw["expected_train_count"])
        if len(records) != expected_train_count:
            raise ValueError("full-train manifest count differs from expected_train_count")
        if {record.partition for record in records} != {"train", "validation"}:
            raise ValueError("full-train manifest has unexpected partitions")
        train_records = records
        validation_records = []
    else:
        train_records = [record for record in records if record.partition == "train"]
        validation_records = [record for record in records if record.partition == "validation"]
    transforms = build_split_transforms(dataset_config.transforms)
    train_dataset = IndexedImageDataset(
        train_records,
        transforms["train"],
        allow_truncated_images=dataset_config.allow_truncated_images,
        stochastic_seed=seed,
    )
    validation_dataset = (
        None
        if training_scope == "full_train_fixed_epochs"
        else IndexedImageDataset(
            validation_records,
            transforms["validation"],
            allow_truncated_images=dataset_config.allow_truncated_images,
        )
    )
    confidence_raw = raw.get("confidence_weighting")
    visual_trust_raw = raw.get("visual_prototype_weighting")
    if confidence_raw is not None and visual_trust_raw is not None:
        raise ValueError("dynamic confidence and visual prototype weighting are separate ablations")
    if label_smoothing > 0.0 and (confidence_raw is not None or visual_trust_raw is not None):
        raise ValueError("label smoothing and sample weighting are separate ablations")
    confidence: ClasswiseHistoryWeighting | None = None
    confidence_warmup_epochs = 0
    if confidence_raw is not None and bool(confidence_raw.get("enabled", True)):
        confidence_warmup_epochs = int(confidence_raw.get("warmup_epochs", 1))
        if confidence_warmup_epochs < 1 or confidence_warmup_epochs >= epochs:
            raise ValueError("confidence warmup_epochs must be in [1, epochs)")
        confidence = ClasswiseHistoryWeighting(
            torch.tensor([record.sample_index for record in train_records]),
            torch.tensor([int(record.internal_class_index) for record in train_records]),
            momentum=float(confidence_raw.get("momentum", 0.8)),
            history_mix=float(confidence_raw.get("history_mix", 0.5)),
            min_weight=float(confidence_raw.get("min_weight", 0.2)),
        )
    visual_trust_lookup: torch.Tensor | None = None
    visual_trust_diagnostics: dict[str, float | int] | None = None
    if visual_trust_raw is not None and bool(visual_trust_raw.get("enabled", True)):
        cache_path = _resolve(base, str(visual_trust_raw["feature_cache"]))
        cache = load_feature_cache(cache_path)
        if cache.metadata.source != "train":
            raise ValueError("visual prototype weighting requires a training feature cache")
        if cache.metadata.manifest_sha256 != _sha256(manifest_path):
            raise ValueError("visual prototype cache manifest hash mismatch")
        if cache.metadata.split_sha256 != _sha256(split_path):
            raise ValueError("visual prototype cache split hash mismatch")
        _, expected_visual_trust_weight_sha256 = _model_settings(model_path)
        if cache.metadata.clip_weight_sha256 != expected_visual_trust_weight_sha256:
            raise ValueError("visual prototype cache CLIP weight hash mismatch")
        if cache.metadata.normalized is not True:
            raise ValueError("visual prototype weighting requires normalized CLIP features")
        cache_positions = {sample_id: index for index, sample_id in enumerate(cache.sample_ids)}
        if len(cache_positions) != len(cache.sample_ids):
            raise ValueError("visual prototype cache contains duplicate sample IDs")
        try:
            positions = torch.tensor([cache_positions[record.sample_id] for record in train_records])
        except KeyError as error:
            raise ValueError("training sample is missing from visual prototype cache") from error
        cached_labels = cache.labels.index_select(0, positions).long()
        record_labels = torch.tensor(
            [int(record.internal_class_index) for record in train_records], dtype=torch.long
        )
        if not torch.equal(cached_labels, record_labels):
            raise ValueError("visual prototype cache labels differ from the training manifest")
        visual_weights, visual_trust_diagnostics = visual_prototype_soft_weights(
            cache.features.index_select(0, positions),
            cached_labels,
            low_trust_fraction=float(visual_trust_raw.get("low_trust_fraction", 0.1)),
            min_weight=float(visual_trust_raw.get("min_weight", 0.5)),
            target_similarity_mix=float(visual_trust_raw.get("target_similarity_mix", 0.5)),
            chunk_size=int(visual_trust_raw.get("chunk_size", 4096)),
        )
        visual_trust_lookup = torch.full(
            (max(record.sample_index for record in train_records) + 1,),
            torch.nan,
            dtype=torch.float32,
        )
        visual_trust_lookup[
            torch.tensor([record.sample_index for record in train_records], dtype=torch.long)
        ] = visual_weights
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
        lora_blocks=int(peft.get("lora_blocks", 1)),
    )
    head, head_checkpoint = load_classifier_head_checkpoint(
        head_checkpoint_path,
        feature_dim=clip_metadata.feature_dim,
        num_classes=num_classes,
        expected_clip_weight_sha256=clip_metadata.weight_sha256,
        expected_class_mapping_sha256=mapping.digest(),
        expected_split_sha256=_sha256(split_path),
        forbid_full_train_initializer=training_scope == "fixed_split_validation",
    )
    head_config = head_checkpoint["head"]
    normalized_features = head_checkpoint["feature_cache_metadata"].get("normalized")
    if not isinstance(normalized_features, bool):
        raise ValueError("head checkpoint must declare whether cached features were normalized")
    head_checkpoint_sha256 = _sha256(head_checkpoint_path)
    model = PEFTImageClassifier(
        visual,
        head,
        normalize_features=normalized_features,
    ).to(device)
    visual_learning_rate = float(training["visual_learning_rate"])
    head_learning_rate = float(training["head_learning_rate"])
    if visual_learning_rate < 0 or head_learning_rate < 0:
        raise ValueError("learning rates must be non-negative")
    if head_learning_rate == 0:
        for parameter in head.parameters():
            parameter.requires_grad_(False)
    visual_parameters = [parameter for parameter in visual.parameters() if parameter.requires_grad]
    head_parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    parameter_groups = []
    if visual_parameters and visual_learning_rate > 0:
        parameter_groups.append({"params": visual_parameters, "lr": visual_learning_rate})
    if head_parameters and head_learning_rate > 0:
        parameter_groups.append({"params": head_parameters, "lr": head_learning_rate})
    if not parameter_groups:
        raise ValueError("training configuration enables no parameters with a positive learning rate")
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=float(training["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    conflict_ids = (
        set()
        if training_scope == "full_train_fixed_epochs"
        else _conflict_ids(manifest_path, conflicts_path)
    )
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
        if confidence is not None:
            if "confidence_weighting" not in state:
                raise ValueError("M1 resume checkpoint is missing confidence state")
            confidence.load_state_dict(state["confidence_weighting"])
        elif "confidence_weighting" in state:
            raise ValueError("non-M1 run cannot resume an M1 confidence checkpoint")

    if epoch == 0 and training_scope == "fixed_split_validation":
        assert validation_dataset is not None
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
                f"raw-image head parity failed: got {initial['all']['top1']}, expected {expected}±{tolerance}"
            )
    else:
        initial = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_started = time.perf_counter()
    while epoch < epochs:
        train_dataset.set_epoch(epoch)
        model.train()
        confidence_active = confidence is not None and epoch >= confidence_warmup_epochs
        applied_confidence = confidence.diagnostics() if confidence is not None else None
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
        epoch_sample_indices: list[torch.Tensor] = []
        epoch_sample_losses: list[torch.Tensor] = []
        epoch_observed_probabilities: list[torch.Tensor] = []
        epoch_predictions: list[torch.Tensor] = []
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
                if confidence_active:
                    batch_weights = confidence.weights_for(batch["sample_index"]).to(
                        device, non_blocking=True
                    )
                    loss = weighted_cross_entropy(logits, labels, batch_weights)
                elif visual_trust_lookup is not None:
                    batch_weights = visual_trust_lookup.index_select(
                        0, batch["sample_index"].long()
                    ).to(device, non_blocking=True)
                    loss = weighted_cross_entropy(logits, labels, batch_weights)
                else:
                    loss = F.cross_entropy(
                        logits,
                        labels,
                        label_smoothing=label_smoothing,
                    )
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
            if confidence is not None:
                detached_logits = logits.detach().float()
                probabilities = detached_logits.softmax(dim=1)
                epoch_sample_indices.append(batch["sample_index"].detach().long().cpu())
                epoch_sample_losses.append(
                    F.cross_entropy(detached_logits, labels, reduction="none").cpu()
                )
                epoch_observed_probabilities.append(
                    probabilities.gather(1, labels[:, None]).squeeze(1).cpu()
                )
                epoch_predictions.append(detached_logits.argmax(dim=1).cpu())
        train_seconds = time.perf_counter() - train_started
        next_confidence = None
        if confidence is not None:
            next_confidence = confidence.update_epoch(
                torch.cat(epoch_sample_indices),
                torch.cat(epoch_sample_losses),
                torch.cat(epoch_observed_probabilities),
                torch.cat(epoch_predictions),
            )
        scheduler.step()
        epoch += 1
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / count,
            "train_seconds": train_seconds,
            "train_samples_per_second": count / train_seconds,
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
        }
        if training_scope == "fixed_split_validation":
            assert validation_dataset is not None
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
            row["validation"] = validation
        if confidence is not None:
            row["confidence_weighting"] = {
                "active": confidence_active,
                "applied": applied_confidence,
                "next": next_confidence,
            }
        history.append(row)
        if training_scope == "fixed_split_validation":
            top1 = float(row["validation"]["all"]["top1"])
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
        if confidence is not None:
            checkpoint["confidence_weighting"] = confidence.state_dict()
        _save_atomic(checkpoint, output / "last.pt")
        if training_scope == "full_train_fixed_epochs":
            print(
                json.dumps(
                    {
                        "stage": "full_train_peft",
                        "status": "epoch_complete",
                        "epoch": epoch,
                        "epochs": epochs,
                        "train_loss": row["train_loss"],
                        "train_seconds": train_seconds,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if training_scope == "full_train_fixed_epochs":
        selected_state = trainable_state_dict(model)
        selected_name = "final.pt"
        selected_top1 = None
        selection_rule = f"fixed final epoch {epochs}; no validation selection"
    else:
        if best_state is None:
            raise RuntimeError("PEFT training produced no best state")
        selected_state = best_state
        selected_name = "best.pt"
        selected_top1 = best_top1
        selection_rule = "best fixed-split validation Top-1"
    _save_atomic(
        {
            "config_sha256": config_sha256,
            "mode": peft["mode"],
            "clip_weight_sha256": clip_metadata.weight_sha256,
            "class_mapping_sha256": mapping.digest(),
            "split_sha256": _sha256(split_path),
            "head_initializer_sha256": head_checkpoint_sha256,
            "head_kind": str(head_config["kind"]),
            "normalize_features": normalized_features,
            "trainable_state": selected_state,
            "best_top1": selected_top1,
            "training_scope": training_scope,
            "selection_rule": selection_rule,
        },
        output / selected_name,
    )
    total_seconds = time.perf_counter() - total_started
    metrics = {
        "scope": (
            "stage1 full official train PEFT final retrain; test data not accessed"
            if training_scope == "full_train_fixed_epochs"
            else "stage1 train/validation PEFT short screening; test data not accessed"
        ),
        "run_id": raw["run_id"],
        "training_scope": training_scope,
        "selection_rule": selection_rule,
        "peft": peft,
        "confidence_weighting": confidence_raw,
        "visual_prototype_weighting": visual_trust_raw,
        "visual_prototype_diagnostics": visual_trust_diagnostics,
        "label_smoothing": label_smoothing,
        "initial_validation": initial,
        "history": history,
        "best_top1": selected_top1,
        "duration_seconds": total_seconds,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0,
        "train_count": len(train_records),
        "validation_count": len(validation_records),
        "total_visual_parameters": summary.total_visual_parameters,
        "trainable_visual_parameters": summary.trainable_visual_parameters,
        "trainable_total_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "selected_checkpoint": selected_name,
        "selected_checkpoint_sha256": _sha256(output / selected_name),
        "last_checkpoint_sha256": _sha256(output / "last.pt"),
        "config_sha256": config_sha256,
        "head_initializer": {
            "path": str(head_checkpoint_path),
            "sha256": head_checkpoint_sha256,
            "kind": str(head_config["kind"]),
            "normalize_features": normalized_features,
        },
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
    if selected_name == "best.pt":
        metrics["best_checkpoint_sha256"] = metrics["selected_checkpoint_sha256"]
    else:
        metrics["final_checkpoint_sha256"] = metrics["selected_checkpoint_sha256"]
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copy2(config_path, output / "config.yaml")
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
