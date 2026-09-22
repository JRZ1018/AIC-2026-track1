"""Benchmark the audited train image pipeline without loading test data or a model."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.contracts import load_dataset_config
from aictrack1.data.indexed_images import IndexedImageDataset, read_indexed_images
from aictrack1.data.transforms import build_split_transforms
from scripts.train_peft import _loader


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_dataset(
    dataset: Any,
    *,
    batch_size: int,
    workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    seed: int,
    warmup_batches: int,
    measure_batches: int,
    repeats: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    if warmup_batches < 0 or measure_batches < 1 or repeats < 1:
        raise ValueError("warmup_batches must be non-negative; measure_batches and repeats must be positive")
    required_batches = warmup_batches + measure_batches
    if len(dataset) < required_batches * batch_size:
        raise ValueError("dataset is too small for the requested non-wrapping benchmark window")

    results: list[dict[str, Any]] = []
    for repeat in range(repeats):
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(0)
        loader_started = time.perf_counter()
        loader = _loader(
            dataset,
            batch_size=batch_size,
            workers=workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            shuffle=True,
            seed=seed,
            device=device,
        )
        iterator = iter(loader)
        loader_setup_seconds = time.perf_counter() - loader_started
        data_wait_seconds = 0.0
        host_to_device_seconds = 0.0
        measured_samples = 0
        _synchronize(device)
        window_started = None
        for batch_index in range(required_batches):
            wait_started = time.perf_counter()
            batch = next(iterator)
            loaded_at = time.perf_counter()
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            _synchronize(device)
            copied_at = time.perf_counter()
            if batch_index >= warmup_batches:
                if window_started is None:
                    window_started = wait_started
                data_wait_seconds += loaded_at - wait_started
                host_to_device_seconds += copied_at - loaded_at
                measured_samples += int(images.shape[0])
        _synchronize(device)
        assert window_started is not None
        window_seconds = time.perf_counter() - window_started
        results.append(
            {
                "repeat": repeat + 1,
                "loader_setup_seconds": loader_setup_seconds,
                "window_seconds": window_seconds,
                "data_wait_seconds": data_wait_seconds,
                "host_to_device_seconds": host_to_device_seconds,
                "measured_batches": measure_batches,
                "measured_samples": measured_samples,
                "samples_per_second": measured_samples / window_seconds,
            }
        )
        del iterator, loader
        gc.collect()
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--partition", choices=("train", "validation"), default="train")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-batches", type=int, default=100)
    parser.add_argument("--measure-batches", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config_path = args.dataset_config.resolve()
    manifest_path = args.manifest.resolve()
    split_path = args.split.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"benchmark output already exists: {output_path}")
    if args.batch_size < 1 or args.workers < 0 or args.prefetch_factor < 1:
        raise ValueError("batch size and prefetch factor must be positive; workers must be non-negative")

    config = load_dataset_config(config_path)
    if str(config.competition_stage) != "stage2":
        raise ValueError("this benchmark is restricted to the Stage 2 dataset contract")
    records = read_indexed_images(
        config,
        source="train",
        manifest_path=manifest_path,
        split_path=split_path,
        partitions=(args.partition,),
    )
    transform = build_split_transforms(config.transforms)[
        "train" if args.partition == "train" else "validation"
    ]
    dataset = IndexedImageDataset(
        records,
        transform,
        allow_truncated_images=config.allow_truncated_images,
        stochastic_seed=args.seed if args.partition == "train" else None,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    repeats = _measure_dataset(
        dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        seed=args.seed,
        warmup_batches=args.warmup_batches,
        measure_batches=args.measure_batches,
        repeats=args.repeats,
        device=device,
    )
    speeds = [float(row["samples_per_second"]) for row in repeats]
    payload = {
        "schema": "aictrack1.image_pipeline_benchmark.v1",
        "scope": "Stage 2 train/validation image pipeline only; test data not accessed",
        "dataset_config": str(config_path),
        "dataset_config_sha256": _sha256(config_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "split": str(split_path),
        "split_sha256": _sha256(split_path),
        "partition": args.partition,
        "record_count": len(records),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "prefetch_factor": args.prefetch_factor if args.workers > 0 else None,
        "persistent_workers": args.persistent_workers if args.workers > 0 else False,
        "warmup_batches": args.warmup_batches,
        "measure_batches": args.measure_batches,
        "seed": args.seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "repeats": repeats,
        "summary": {
            "samples_per_second_min": min(speeds),
            "samples_per_second_max": max(speeds),
            "samples_per_second_mean": sum(speeds) / len(speeds),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
