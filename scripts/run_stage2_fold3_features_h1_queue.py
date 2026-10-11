"""Prepare independent 224/336 frozen H1s for the pre-registered Stage 2 fold 3."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import TrainingLease  # noqa: E402

QUEUE = ROOT / "artifacts/queues/20260925-stage2-fold3-features-h1"
MANIFEST = ROOT / "artifacts/audits/stage2/train/samples.csv"
SPLIT = ROOT / "artifacts/audits/stage2/train/split_fold3_seeded_random_20260925.csv"
MODEL = ROOT / "configs/model/openai_clip_vit_b32.yaml"
LOCK = ROOT / "artifacts/queues/training.lock.json"
DEADLINE = datetime(2026, 9, 25, 19, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
CASES = (
    (224, "c3", "stage2_fold3_224.yaml", "stage2_fold3_c3_h1_224_seed20260921.yaml"),
    (336, "c5", "stage2_fold3_336.yaml", "stage2_fold3_c5_h1_336_seed20260921.yaml"),
)


def log(message: str) -> None:
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    with (QUEUE / "chain.log").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"[{stamp}] {message}\n")


def run_step(lease: TrainingLease, tag: str, command: list[str]) -> None:
    if datetime.now(ZoneInfo("Asia/Shanghai")) >= DEADLINE:
        raise TimeoutError(f"budget deadline reached before {tag}")
    log(f"{tag}: start")
    with (QUEUE / f"{tag}.log").open("w", encoding="utf-8", newline="\n") as handle:
        child = subprocess.Popen(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        lease.attach_trainer(child.pid)
        while child.poll() is None:
            if datetime.now(ZoneInfo("Asia/Shanghai")) >= DEADLINE:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(f"{tag} child termination unconfirmed; retain lock") from exc
                raise TimeoutError(f"budget deadline stopped {tag}; inspect shards/checkpoint before resume")
            time.sleep(5)
        if child.returncode != 0:
            raise RuntimeError(f"{tag} failed with code {child.returncode}; inspect {tag}.log")
    log(f"{tag}: complete")


def main() -> int:
    if QUEUE.exists():
        raise FileExistsError(QUEUE)
    for path in (MANIFEST, SPLIT, MODEL):
        if not path.is_file():
            raise FileNotFoundError(path)
    for resolution, tag, data_name, h1_name in CASES:
        paths = (
            ROOT / "configs/data" / data_name,
            ROOT / "configs/experiment" / h1_name,
        )
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(paths)
        feature_dir = ROOT / f"artifacts/features/stage2/train_openai_vit_b32_fold3_{resolution}_20260925"
        h1_dir = ROOT / f"artifacts/experiments/20260925-stage2-fold3-{tag}-h1-{resolution}-seed20260921"
        if feature_dir.exists() or h1_dir.exists():
            raise FileExistsError(f"fresh fold-3 output already exists: {feature_dir} / {h1_dir}")
    lease = TrainingLease.acquire(
        LOCK, owner="20260925-stage2-fold3-features-h1",
        config=str(ROOT / "configs/data/stage2_fold3_224.yaml"),
        output_dir=str(QUEUE), command=tuple(sys.argv),
    )
    try:
        QUEUE.mkdir(parents=True, exist_ok=False)
        log("Stage2 train-only fold3 224 and 336 feature/H1 queue started")
        for resolution, tag, data_name, h1_name in CASES:
            feature_dir = ROOT / f"artifacts/features/stage2/train_openai_vit_b32_fold3_{resolution}_20260925"
            h1_dir = ROOT / f"artifacts/experiments/20260925-stage2-fold3-{tag}-h1-{resolution}-seed20260921"
            run_step(lease, f"features-{resolution}", [
                sys.executable, str(ROOT / "scripts/extract_clip_features.py"),
                "--config", str(ROOT / "configs/data" / data_name),
                "--source", "train", "--manifest", str(MANIFEST), "--split", str(SPLIT),
                "--model-config", str(MODEL), "--visual-resolution", str(resolution),
                "--device", "cuda", "--feature-dtype", "float16", "--output", str(feature_dir),
                "--batch-size", "64", "--workers", "4", "--shard-size", "4096",
            ])
            stats = json.loads((feature_dir / "extraction_stats.json").read_text(encoding="utf-8"))
            if stats["record_count"] != 148695 or not (feature_dir / "complete.json").is_file():
                raise RuntimeError(f"{resolution} feature cache incomplete")
            run_step(lease, f"h1-{resolution}", [
                sys.executable, str(ROOT / "scripts/train_frozen_head.py"),
                "--config", str(ROOT / "configs/experiment" / h1_name),
            ])
            metrics = json.loads((h1_dir / "metrics.json").read_text(encoding="utf-8"))
            if len(metrics["history"]) != 100 or not (h1_dir / "best.pt").is_file():
                raise RuntimeError(f"{resolution} H1 incomplete")
            log(f"h1-{resolution}: best_top1={metrics['best_top1']}")
        log("fold3 features/H1 complete; freeze H1 initial-accuracy checks before PEFT")
    except Exception as exc:
        if QUEUE.exists():
            log(f"queue failed: {type(exc).__name__}: {exc}")
        raise
    finally:
        lease.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
