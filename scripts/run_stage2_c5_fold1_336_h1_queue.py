"""Extract Stage 2 train-only 336 features and train the fixed old-fold H1.

The PEFT comparison is deliberately a later queue: its expected initial
accuracy must be frozen from this H1's actual metrics before launch.
"""

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

QUEUE = ROOT / "artifacts" / "queues" / "20260925-stage2-c5-fold1-336-h1"
FEATURES = ROOT / "artifacts" / "features" / "stage2" / "train_openai_vit_b32_fold1_336_20260925"
H1 = ROOT / "artifacts" / "experiments" / "20260925-stage2-fold1-c5-h1-336-seed20260921"
DATA = ROOT / "configs" / "data" / "stage2_fold1_336.yaml"
H1_CONFIG = ROOT / "configs" / "experiment" / "stage2_fold1_c5_h1_336_seed20260921.yaml"
MANIFEST = ROOT / "artifacts" / "audits" / "stage2" / "train" / "samples.csv"
SPLIT = ROOT / "artifacts" / "audits" / "stage2" / "train" / "split.csv"
MODEL_CONFIG = ROOT / "configs" / "model" / "openai_clip_vit_b32.yaml"
DEADLINE = datetime(2026, 9, 25, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


def log(message: str) -> None:
    with (QUEUE / "chain.log").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"[{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}] {message}\n")


def run_step(lease: TrainingLease, tag: str, command: list[str]) -> None:
    if datetime.now(ZoneInfo("Asia/Shanghai")) >= DEADLINE:
        raise TimeoutError("09:00 deadline reached before launch")
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
                child.wait(timeout=30)
                raise TimeoutError(f"stopped {tag} at 09:00")
            time.sleep(5)
        if child.returncode != 0:
            raise RuntimeError(f"{tag} failed with exit code {child.returncode}")
    log(f"{tag}: complete")


def main() -> int:
    for path in (QUEUE, FEATURES, H1):
        if path.exists():
            raise FileExistsError(f"fresh old-fold 336 H1 queue refuses existing output: {path}")
    for path in (DATA, H1_CONFIG, MANIFEST, SPLIT, MODEL_CONFIG):
        if not path.is_file():
            raise FileNotFoundError(path)
    lease = TrainingLease.acquire(
        ROOT / "artifacts" / "queues" / "training.lock.json",
        owner="20260925-stage2-c5-fold1-336-h1",
        config=str(H1_CONFIG), output_dir=str(H1), command=tuple(sys.argv),
    )
    try:
        QUEUE.mkdir(parents=True, exist_ok=False)
        log("old-fold 336 train-only feature and H1 queue started")
        run_step(lease, "features", [
            sys.executable, str(ROOT / "scripts" / "extract_clip_features.py"),
            "--config", str(DATA), "--source", "train", "--manifest", str(MANIFEST),
            "--split", str(SPLIT), "--model-config", str(MODEL_CONFIG),
            "--visual-resolution", "336", "--device", "cuda", "--feature-dtype", "float16",
            "--output", str(FEATURES), "--batch-size", "64", "--workers", "4", "--shard-size", "4096",
        ])
        stats = json.loads((FEATURES / "extraction_stats.json").read_text(encoding="utf-8"))
        if stats["record_count"] != 148695:
            raise RuntimeError("old-fold 336 feature count is incomplete")
        run_step(lease, "h1", [
            sys.executable, str(ROOT / "scripts" / "train_frozen_head.py"),
            "--config", str(H1_CONFIG),
        ])
        metrics = json.loads((H1 / "metrics.json").read_text(encoding="utf-8"))
        if not (H1 / "best.pt").is_file() or len(metrics["history"]) != 100:
            raise RuntimeError("old-fold H1 is incomplete")
        log("old-fold 336 H1 completed; freeze expected_initial_top1 before PEFT")
    except Exception as exc:
        log(f"queue failed: {type(exc).__name__}: {exc}")
        raise
    finally:
        lease.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
