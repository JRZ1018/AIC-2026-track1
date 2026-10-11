"""Atomic local queue for the preregistered matched-336 Stage 2 fold-2 C5.

Only official Stage 2 train is read. The queue stops before 09:00 local time;
each PEFT epoch has its own recoverable checkpoint.
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

QUEUE = ROOT / "artifacts" / "queues" / "20260925-stage2-c5-336-local"
FEATURES = ROOT / "artifacts" / "features" / "stage2" / "train_openai_vit_b32_fold2_336_20260925"
SMOKE = ROOT / "artifacts" / "features" / "stage2" / "train_openai_vit_b32_fold2_336_smoke_20260925"
H1 = ROOT / "artifacts" / "experiments" / "20260925-stage2-fold2-c5-h1-336-seed20260921"
PEFT = ROOT / "artifacts" / "experiments" / "20260925-stage2-fold2-c5-336-last8-ac-8ep-seed20260921"
DATA = ROOT / "configs" / "data" / "stage2_fold2_336.yaml"
H1_CONFIG = ROOT / "configs" / "experiment" / "stage2_fold2_c5_h1_336_seed20260921.yaml"
PEFT_CONFIG = ROOT / "configs" / "experiment" / "stage2_fold2_c5_336_last8_ac_8ep_seed20260921.yaml"
MANIFEST = ROOT / "artifacts" / "audits" / "stage2" / "train" / "samples.csv"
SPLIT = ROOT / "artifacts" / "audits" / "stage2" / "train" / "split_fold2_seeded_random_20260924.csv"
MODEL_CONFIG = ROOT / "configs" / "model" / "openai_clip_vit_b32.yaml"
DEADLINE = datetime(2026, 9, 25, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


def now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Shanghai"))


def write_log(message: str) -> None:
    with (QUEUE / "chain.log").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"[{now().isoformat()}] {message}\n")


def run_step(lease: TrainingLease, tag: str, command: list[str]) -> None:
    if now() >= DEADLINE:
        raise TimeoutError(f"deadline reached before {tag}")
    write_log(f"{tag}: start")
    with (QUEUE / f"{tag}.log").open("w", encoding="utf-8", newline="\n") as handle:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        lease.attach_trainer(child.pid)
        while child.poll() is None:
            if now() >= DEADLINE:
                child.terminate()
                child.wait(timeout=30)
                raise TimeoutError(f"deadline stopped {tag}; inspect its complete last.pt before resume")
            time.sleep(5)
        if child.returncode != 0:
            raise RuntimeError(f"{tag} failed with exit code {child.returncode}; see {tag}.log")
    write_log(f"{tag}: complete")


def main() -> int:
    for path in (SMOKE, FEATURES, H1, PEFT, QUEUE):
        if path.exists():
            raise FileExistsError(f"fresh C5 queue refuses existing output: {path}")
    for path in (DATA, H1_CONFIG, PEFT_CONFIG, MANIFEST, SPLIT, MODEL_CONFIG):
        if not path.is_file():
            raise FileNotFoundError(path)
    lease = TrainingLease.acquire(
        ROOT / "artifacts" / "queues" / "training.lock.json",
        owner="20260925-stage2-c5-336-local",
        config=str(PEFT_CONFIG),
        output_dir=str(PEFT),
        command=tuple(sys.argv),
    )
    QUEUE.mkdir(parents=True, exist_ok=False)
    write_log("C5 matched-336 fold-2 queue started; official Stage2 train only")
    try:
        common = [
            sys.executable,
            str(ROOT / "scripts" / "extract_clip_features.py"),
            "--config", str(DATA),
            "--source", "train",
            "--manifest", str(MANIFEST),
            "--split", str(SPLIT),
            "--model-config", str(MODEL_CONFIG),
            "--visual-resolution", "336",
            "--device", "cuda",
            "--feature-dtype", "float16",
        ]
        run_step(
            lease, "features-smoke",
            common + ["--output", str(SMOKE), "--batch-size", "8", "--workers", "0", "--max-samples", "32"],
        )
        if not (SMOKE / "extraction_stats.json").is_file():
            raise RuntimeError("336 feature smoke did not finalize")
        run_step(
            lease, "features-full",
            common + ["--output", str(FEATURES), "--batch-size", "64", "--workers", "4", "--shard-size", "4096"],
        )
        feature_stats = json.loads((FEATURES / "extraction_stats.json").read_text(encoding="utf-8"))
        if feature_stats["record_count"] != 148695:
            raise RuntimeError("336 feature cache is incomplete")
        run_step(
            lease, "h1-336",
            [sys.executable, str(ROOT / "scripts" / "train_frozen_head.py"), "--config", str(H1_CONFIG)],
        )
        if not (H1 / "best.pt").is_file():
            raise RuntimeError("336 H1 did not save best.pt")
        run_step(
            lease, "peft-336-fold2",
            [sys.executable, str(ROOT / "scripts" / "train_peft.py"), "--config", str(PEFT_CONFIG)],
        )
        metrics = json.loads((PEFT / "metrics.json").read_text(encoding="utf-8"))
        if metrics["train_count"] != 118958 or metrics["validation_count"] != 29737 or len(metrics["history"]) != 8:
            raise RuntimeError("C5 metrics do not contain the complete fixed fold")
        if not (PEFT / "best.pt").is_file():
            raise RuntimeError("C5 best.pt missing")
        write_log("C5 queue completed; compare epoch-8 preregistered gate before any next task")
    except Exception as exc:
        write_log(f"C5 queue failed: {type(exc).__name__}: {exc}")
        raise
    finally:
        lease.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
