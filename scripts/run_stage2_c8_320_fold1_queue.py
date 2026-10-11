"""Run the preregistered fold-1 320-resolution screening chain under the training lock."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import TrainingLease  # noqa: E402

QUEUE = ROOT / "artifacts/queues/20260926-stage2-c8-320-fold1"
DATA = ROOT / "configs/data/stage2_fold1_320.yaml"
H1_CONFIG = ROOT / "configs/experiment/stage2_fold1_c8_h1_320_seed20260921.yaml"
PEFT_CONFIG = ROOT / "configs/experiment/stage2_fold1_c8_320_last8_ac_8ep_seed20260921.yaml"
MANIFEST = ROOT / "artifacts/audits/stage2/train/samples.csv"
SPLIT = ROOT / "artifacts/audits/stage2/train/split.csv"
MODEL = ROOT / "configs/model/openai_clip_vit_b32.yaml"
FEATURES = ROOT / "artifacts/features/stage2/train_openai_vit_b32_fold1_320_20260926"
H1 = ROOT / "artifacts/experiments/20260926-stage2-fold1-c8-h1-320-seed20260921"
PEFT = ROOT / "artifacts/experiments/20260926-stage2-fold1-c8-320-last8-ac-8ep-seed20260921"
DEADLINE = datetime.now(ZoneInfo("Asia/Shanghai")) + timedelta(hours=12)


def log(message: str) -> None:
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    with (QUEUE / "chain.log").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"[{stamp}] {message}\n")


def run_step(lease: TrainingLease, tag: str, command: list[str]) -> None:
    if datetime.now(ZoneInfo("Asia/Shanghai")) >= DEADLINE:
        raise TimeoutError(f"approved 12-hour deadline reached before {tag}")
    log(f"{tag}: start {' '.join(command)}")
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
            if datetime.now(ZoneInfo("Asia/Shanghai")) >= DEADLINE:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(f"{tag} termination unconfirmed; retain lock") from exc
                raise TimeoutError(f"approved 12-hour deadline stopped {tag}")
            time.sleep(10)
        if child.returncode != 0:
            raise RuntimeError(f"{tag} failed with exit code {child.returncode}; inspect {QUEUE / (tag + '.log')}")
    log(f"{tag}: complete")


def main() -> int:
    for path in (QUEUE, FEATURES, H1, PEFT):
        if path.exists():
            raise FileExistsError(f"fresh 320 screening output already exists: {path}")
    for path in (DATA, H1_CONFIG, PEFT_CONFIG, MANIFEST, SPLIT, MODEL):
        if not path.is_file():
            raise FileNotFoundError(path)
    lease = TrainingLease.acquire(
        ROOT / "artifacts/queues/training.lock.json",
        owner="20260926-stage2-c8-320-fold1",
        config=str(PEFT_CONFIG),
        output_dir=str(QUEUE),
        command=tuple(sys.argv),
    )
    try:
        QUEUE.mkdir(parents=True, exist_ok=False)
        log("320 fold1 screening chain started; train-only, no test access")
        run_step(lease, "features-320", [
            sys.executable, str(ROOT / "scripts/extract_clip_features.py"),
            "--config", str(DATA), "--source", "train", "--manifest", str(MANIFEST),
            "--split", str(SPLIT), "--model-config", str(MODEL),
            "--visual-resolution", "320", "--device", "cuda", "--feature-dtype", "float16",
            "--output", str(FEATURES), "--batch-size", "64", "--workers", "4", "--shard-size", "4096",
        ])
        stats = json.loads((FEATURES / "extraction_stats.json").read_text(encoding="utf-8"))
        if stats["record_count"] != 148695 or not (FEATURES / "complete.json").is_file():
            raise RuntimeError("320 feature cache is incomplete")
        run_step(lease, "h1-320", [sys.executable, str(ROOT / "scripts/train_frozen_head.py"), "--config", str(H1_CONFIG)])
        metrics = json.loads((H1 / "metrics.json").read_text(encoding="utf-8"))
        if len(metrics["history"]) != 100 or not (H1 / "best.pt").is_file():
            raise RuntimeError("320 H1 is incomplete")
        log(f"h1-320: best_top1={metrics['best_top1']}")
        run_step(lease, "peft-320", [sys.executable, str(ROOT / "scripts/train_peft.py"), "--config", str(PEFT_CONFIG)])
        peft_metrics = json.loads((PEFT / "metrics.json").read_text(encoding="utf-8"))
        if len(peft_metrics["history"]) != 8 or not (PEFT / "best.pt").is_file():
            raise RuntimeError("320 PEFT is incomplete")
        log(f"peft-320: best_top1={peft_metrics.get('best_top1')}")
        log("320 fold1 screening chain completed")
    except Exception as exc:
        if QUEUE.exists():
            log(f"queue failed: {type(exc).__name__}: {exc}")
        raise
    finally:
        lease.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
