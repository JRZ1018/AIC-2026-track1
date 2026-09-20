"""Run the second follow-up queue: the strongest levers and a seed lottery.

Derived entirely from item S, the 61.3690 best (full data, frozen full head, 12 epochs,
mix 0.75, min_weight 0.05, seed 20260910):

  AA  min_weight 0.05 with 16 epochs   - the two strongest known levers together
  AB  seed 20260912 on the best recipe - a draw from the new best distribution
  AC  min_weight 0.02 with 16 epochs   - floor pushed further while extending duration

The queue uses the shared atomic training lock and fail-closed process probe. It never
continues past a partial output or an ambiguously terminated trainer.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import QueueItem, run_serial_queue  # noqa: E402

PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
LOGS = ROOT / "artifacts" / "queues" / "20260919-batch2"

D = "configs/experiment/"
E = "artifacts/experiments/"

QUEUE = [
    ("AA", D + "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw05_magmix75_20ep_full_seed1_aa.yaml",
     E + "20260919-aa-final-h1init-t2-last4-lr175e6-cwh-mw05-magmix75-20ep-full-seed1",
     "mw 0.05 with 20 epochs: best floor pushed to the longest duration"),
    ("AB", D + "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw05_magmix75_12ep_full_seed2_ab.yaml",
     E + "20260919-ab-final-h1init-t2-last4-lr175e6-cwh-mw05-magmix75-12ep-full-seed2",
     "seed lottery on the 61.3690 recipe"),
    ("AC", D + "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw02_magmix75_16ep_full_seed1_ac.yaml",
     E + "20260919-ac-final-h1init-t2-last4-lr175e6-cwh-mw02-magmix75-16ep-full-seed1",
     "min_weight 0.02 with 16 epochs"),
]


def main() -> int:
    return run_serial_queue(
        root=ROOT,
        python=PYTHON,
        logs=LOGS,
        queue_name="20260919-batch2",
        items=[QueueItem(*item) for item in QUEUE],
    )


if __name__ == "__main__":
    raise SystemExit(main())
