"""Train a reproducible CE linear or cosine head from frozen CLIP features."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.feature_cache import load_feature_cache, select_feature_partition
from aictrack1.data.long_tail_split import (
    LONG_TAIL_TRAIN,
    read_split_assignments,
    select_balanced_size_matched_ids,
    select_partition_indices,
)
from aictrack1.engine.frozen_head import FrozenHeadConfig, FrozenHeadTrainer, FrozenLossConfig
from aictrack1.models.heads import build_classifier_head

STRESS_SCOPES = frozenset(
    {"long_tail_stress", "balanced_size_matched_control", "balanced_full_train_control"}
)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw = raw.get("experiment", raw)
    base = config_path.parent
    cache_path = _resolve(base, str(raw["feature_cache"]))
    output = _resolve(base, str(raw["output_dir"]))
    if output.exists() and args.resume is None:
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    training_scope = str(raw.get("training_scope", "fixed_split_validation"))
    if training_scope not in {"fixed_split_validation", "full_train_fixed_epochs"} | STRESS_SCOPES:
        raise ValueError(f"unsupported frozen-head training scope {training_scope!r}")
    if training_scope == "full_train_fixed_epochs":
        print(
            json.dumps(
                {
                    "stage": "full_train_head",
                    "status": "loading_feature_cache",
                    "feature_cache": str(cache_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    cache = load_feature_cache(cache_path)
    if cache.metadata.source != "train":
        raise ValueError("frozen-head training refuses test feature caches")
    if training_scope == "full_train_fixed_epochs":
        expected_train_count = int(raw["expected_train_count"])
        if cache.metadata.expected_record_count != expected_train_count:
            raise ValueError("full-train feature cache count differs from expected_train_count")
        if set(cache.partitions) != {"train", "validation"}:
            raise ValueError("full-train feature cache has unexpected partitions")
        train_features = cache.features.float()
        train_labels = cache.labels
        if (train_labels < 0).any():
            raise ValueError("full-train feature cache contains missing labels")
        validation_features = validation_labels = None
    else:
        train_features, train_labels = select_feature_partition(cache, "train")
        validation_features, validation_labels = select_feature_partition(cache, "validation")
    if training_scope in STRESS_SCOPES:
        # Migration stress training reuses the frozen train-partition features but
        # restricts them to the deterministic long-tail (or size-matched) subset.
        long_tail_split_path = _resolve(base, str(raw["long_tail_split"]))
        assignment = read_split_assignments(long_tail_split_path, column="long_tail_split")
        train_sample_ids = list(cache.sample_ids)
        if len(set(train_sample_ids)) != len(train_sample_ids):
            raise ValueError("feature cache contains duplicate sample IDs")
        train_labels_all = [int(value) for value in cache.labels.tolist()]
        long_tail_rows = select_partition_indices(
            train_sample_ids,
            assignment,
            partition=LONG_TAIL_TRAIN,
            available=set(train_sample_ids),
        )
        cache_index = {sample_id: index for index, sample_id in enumerate(train_sample_ids)}
        if training_scope == "balanced_size_matched_control":
            balanced_ids = select_balanced_size_matched_ids(
                [train_sample_ids[position] for position in long_tail_rows],
                [train_labels_all[position] for position in long_tail_rows],
                total=len(long_tail_rows),
                seed=int(raw["training"]["seed"]),
                num_classes=int(raw["num_classes"]),
            )
            selected = [cache_index[sample_id] for sample_id in balanced_ids]
        elif training_scope == "balanced_full_train_control":
            # Same sample budget as the size-matched control, but drawn from the whole
            # frozen train partition so the comparison varies label skew alone.
            target_total = int(raw.get("expected_train_count", len(long_tail_rows)))
            balanced_pool = [
                position
                for position, partition in enumerate(cache.partitions)
                if partition == "train"
            ]
            balanced_ids = select_balanced_size_matched_ids(
                [train_sample_ids[position] for position in balanced_pool],
                [train_labels_all[position] for position in balanced_pool],
                total=target_total,
                seed=int(raw["training"]["seed"]),
                num_classes=int(raw["num_classes"]),
            )
            selected = [cache_index[sample_id] for sample_id in balanced_ids]
        else:
            selected = long_tail_rows
        selected = sorted(selected)
        index = torch.tensor(selected, dtype=torch.long)
        train_features = cache.features.index_select(0, index).float()
        train_labels = cache.labels.index_select(0, index)
        validation_features = validation_labels = None
        print(
            json.dumps(
                {
                    "stage": training_scope,
                    "status": "selected_stress_subset",
                    "record_count": len(selected),
                    "long_tail_split_sha256": _sha256(long_tail_split_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    num_classes = int(raw["num_classes"])
    if int(train_labels.max()) >= num_classes or (
        validation_labels is not None and int(validation_labels.max()) >= num_classes
    ):
        raise ValueError("feature labels exceed configured num_classes")

    training = FrozenHeadConfig(**raw["training"])
    loss_config = FrozenLossConfig(**raw.get("loss", {}))
    random.seed(training.seed)
    np.random.seed(training.seed)
    torch.manual_seed(training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training.seed)
    head_raw = raw["head"]
    head = build_classifier_head(
        str(head_raw["kind"]),
        cache.metadata.feature_dim,
        num_classes,
        cosine_initial_scale=float(head_raw.get("initial_scale", 10.0)),
        cosine_learnable_scale=bool(head_raw.get("learnable_scale", True)),
    )
    device = torch.device(str(raw.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    trainer = FrozenHeadTrainer(
        head,
        training,
        device=device,
        num_classes=num_classes,
        loss_config=loss_config,
    )
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    if training_scope == "full_train_fixed_epochs" or training_scope in STRESS_SCOPES:
        history = trainer.history
        while trainer.epoch < training.epochs:
            train_loss = trainer.train_epoch(train_features, train_labels)
            history.append(
                {
                    "epoch": trainer.epoch,
                    "train_loss": train_loss,
                    "learning_rate": float(trainer.optimizer.param_groups[0]["lr"]),
                }
            )
            trainer.save_checkpoint(output / "last.pt")
            print(
                json.dumps(
                    {
                        "stage": "full_train_head",
                        "status": "epoch_complete",
                        "epoch": trainer.epoch,
                        "epochs": training.epochs,
                        "train_loss": train_loss,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    else:
        assert validation_features is not None and validation_labels is not None
        history = trainer.fit(
            train_features, train_labels, validation_features, validation_labels
        )
    duration = time.perf_counter() - started
    trainer.save_checkpoint(output / "last.pt")
    if training_scope == "full_train_fixed_epochs" or training_scope in STRESS_SCOPES:
        selected_state = trainer.model.state_dict()
        selected_name = "final.pt"
        selected_top1 = None
        selection_rule = f"fixed final epoch {training.epochs}; no validation selection"
    else:
        if trainer.best_state is None:
            raise RuntimeError("training did not produce a best checkpoint")
        selected_state = trainer.best_state
        selected_name = "best.pt"
        selected_top1 = trainer.best_top1
        selection_rule = "best fixed-split validation Top-1"
    feature_metadata = cache.metadata.__dict__ | {
        "training_scope": training_scope,
        "training_record_count": len(train_labels),
    }
    if training_scope in STRESS_SCOPES:
        # Record the exact per-class training support so evaluations can probe
        # starvation without guessing which rows the run consumed.
        support = torch.bincount(train_labels.long(), minlength=num_classes)
        feature_metadata["training_class_support"] = [int(value) for value in support.tolist()]
    torch.save(
        {
            "model": selected_state,
            "head": head_raw,
            "loss": loss_config.__dict__,
            "feature_cache_metadata": feature_metadata,
            "best_top1": selected_top1,
            "selection_rule": selection_rule,
        },
        output / selected_name,
    )
    scope_descriptions = {
        "full_train_fixed_epochs": (
            "stage1 full official train frozen-feature final retrain; test data not accessed"
        ),
        "long_tail_stress": (
            "stage1 long-tail stress frozen-feature head; current-stage train partition only; "
            "test data not accessed"
        ),
        "balanced_size_matched_control": (
            "stage1 class-balanced size-matched frozen-feature control; current-stage train "
            "partition only; test data not accessed"
        ),
        "balanced_full_train_control": (
            "stage1 class-balanced full-train frozen-feature control; current-stage train "
            "partition only; test data not accessed"
        ),
    }
    metrics = {
        "scope": scope_descriptions.get(
            training_scope, "stage1 noisy-validation frozen-feature baseline"
        ),
        "training_scope": training_scope,
        "selection_rule": selection_rule,
        "history": history,
        "loss": loss_config.__dict__,
        "best_top1": selected_top1,
        "duration_seconds": duration,
        "peak_vram_mib": (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
        ),
        "train_count": len(train_labels),
        "validation_count": 0 if validation_labels is None else len(validation_labels),
        "feature_cache": str(cache_path),
        "feature_cache_metadata": feature_metadata,
        "selected_checkpoint": selected_name,
        "selected_checkpoint_sha256": _sha256(output / selected_name),
        "last_checkpoint_sha256": _sha256(output / "last.pt"),
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
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    (output / "environment.json").write_text(
        json.dumps(
            {
                "git_commit": git,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
