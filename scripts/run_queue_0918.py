"""Run the 2026-09-18 follow-up queue back to back, one full train at a time.

Order and purpose:
  P  mix 0.75 -> 1.00        extends the coefficient axis, which gained twice already
  R  seed 20260910 -> 20260912  the run-to-run noise scale, without which no single
                                online delta can be judged
  S  min_weight 0.10 -> 0.05  the other end of the same aggressiveness lever
  T  12 -> 16 epochs          duration, the axis with the strongest online evidence

Each run skips its training step if its output directory already holds metrics.json, so a
restart after an interruption resumes the queue instead of repeating finished work. Any
run that exits without metrics.json is recorded and the queue stops, rather than pressing
on and producing an unreadable chain.

Usage:  .\.venv\Scripts\python.exe scripts\run_queue_0918.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
LOGS = ROOT / "artifacts" / "queues" / "20260918-batch6b"

QUEUE = [
    ("P", "configs/experiment/p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix100_12ep_full_seed1_p.yaml",
     "artifacts/experiments/20260918-p-final-h1init-t2-last4-lr175e6-cwh-mw10-magmix100-12ep-full-seed1",
     "mix 0.75 -> 1.00 (coefficient axis endpoint)"),
    ("R", "configs/experiment/p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix75_12ep_full_seed2_r.yaml",
     "artifacts/experiments/20260918-r-final-h1init-t2-last4-lr175e6-cwh-mw10-magmix75-12ep-full-seed2",
     "seed 20260910 -> 20260912 (online noise scale)"),
    ("S", "configs/experiment/p3_stage1_h1init_t2_last4_lr175e6_cwh_mw05_magmix75_12ep_full_seed1_s.yaml",
     "artifacts/experiments/20260918-s-final-h1init-t2-last4-lr175e6-cwh-mw05-magmix75-12ep-full-seed1",
     "min_weight 0.10 -> 0.05 (same lever, stronger)"),
    ("T", "configs/experiment/p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix75_16ep_full_seed1_t.yaml",
     "artifacts/experiments/20260918-t-final-h1init-t2-last4-lr175e6-cwh-mw10-magmix75-16ep-full-seed1",
     "12 -> 16 epochs (duration)"),
]


def _log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with (LOGS / "chain.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    _log("queue start")
    for tag, config, output, purpose in QUEUE:
        metrics = ROOT / output / "metrics.json"
        if metrics.is_file():
            _log(f"{tag}: already complete, skipping ({purpose})")
            continue
        log_path = LOGS / f"{tag}.log"
        _log(f"{tag}: starting - {purpose}")
        started = time.time()
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [str(PYTHON), "scripts/train_peft.py", "--config", config],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        elapsed = time.time() - started
        if metrics.is_file():
            _log(f"{tag}: finished in {elapsed / 60:.1f} min, metrics.json written")
        else:
            _log(
                f"{tag}: STOPPED after {elapsed / 60:.1f} min with return code "
                f"{completed.returncode} and NO metrics.json; halting the queue. "
                f"Log: {log_path}"
            )
            return 1
    _log("queue finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
