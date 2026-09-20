"""Run the 2026-09-18 follow-up queue back to back, one full train at a time.
Order and purpose:
  P  mix 0.75 -> 1.00        extends the coefficient axis, which gained twice already
  R  seed 20260910 -> 20260912  the run-to-run noise scale, without which no single
                                online delta can be judged
  S  min_weight 0.10 -> 0.05  the other end of the same aggressiveness lever
  T  12 -> 16 epochs          duration, the axis with the strongest online evidence

All historical items now use the shared atomic training lock. A run is skipped only when
its metrics, selected checkpoint, config, and command are all present. Partial outputs,
an external trainer, an unreadable process probe, or a stale lock stop the queue.

Usage:  run it with the repository virtual environment's python, passing this file path.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import QueueItem, run_serial_queue  # noqa: E402

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


def main() -> int:
    return run_serial_queue(
        root=ROOT,
        python=PYTHON,
        logs=LOGS,
        queue_name="20260918-batch6b",
        items=[QueueItem(*item) for item in QUEUE],
    )


if __name__ == "__main__":
    raise SystemExit(main())
