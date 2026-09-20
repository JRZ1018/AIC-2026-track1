"""Batch 3 queue: continue the duration axis and draw more seeds from the best recipes.

By this point the measured facts are: the online seed noise is +-0.164, min_weight 0.10 ->
0.05 at 12 epochs was worth +0.3685 (above the noise), and training loss kept falling through
20 epochs (0.2324 at 12, 0.2009 at 16, 0.1740 at 20), so duration is the one lever whose
historical effect dwarfs the noise.

  AD  20 epochs, seed 20260912  - a second draw from the longest-duration recipe
  AE  min_weight 0.05 at 24 epochs - best floor with duration pushed further still

The queue uses the shared atomic training lock and fail-closed process probe.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import QueueItem, run_serial_queue  # noqa: E402

PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
LOGS = ROOT / "artifacts" / "queues" / "20260919-batch3"

D = "configs/experiment/"
E = "artifacts/experiments/"

QUEUE = [
    ("AD", D + "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix75_20ep_full_seed2_ad.yaml",
     E + "20260919-ad-final-h1init-t2-last4-lr175e6-cwh-mw10-magmix75-20ep-full-seed2",
     "20 epochs at a fresh seed: second draw from the longest-duration recipe"),
    ("AE", D + "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw05_magmix75_24ep_full_seed1_ae.yaml",
     E + "20260919-ae-final-h1init-t2-last4-lr175e6-cwh-mw05-magmix75-24ep-full-seed1",
     "min_weight 0.05 with 24 epochs: best floor, longest duration"),
]


def main() -> int:
    return run_serial_queue(
        root=ROOT,
        python=PYTHON,
        logs=LOGS,
        queue_name="20260919-batch3",
        items=[QueueItem(*item) for item in QUEUE],
    )


if __name__ == "__main__":
    raise SystemExit(main())
