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
from aictrack1.engine.frozen_head import FrozenHeadConfig, FrozenHeadTrainer, FrozenLossConfig
from aictrack1.models.heads import build_classifier_head


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
    if training_scope not in {"fixed_split_validation", "full_train_fixed_epochs"}:
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
    if training_scope == "full_train_fixed_epochs":
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
    if training_scope == "full_train_fixed_epochs":
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
    metrics = {
        "scope": (
            "stage1 full official train frozen-feature final retrain; test data not accessed"
            if training_scope == "full_train_fixed_epochs"
            else "stage1 noisy-validation frozen-feature baseline"
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
