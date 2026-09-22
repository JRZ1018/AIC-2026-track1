"""Run the two unlocked Stage2 R4 AC tail-protection screens serially."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import QueueItem, run_serial_queue  # noqa: E402


LOGS = ROOT / "artifacts" / "queues" / "20260922-stage2-r4-ac"
QUEUE = [
    QueueItem(
        "R4-4",
        "configs/experiment/stage2_r4_ac_tailmeanfloor035_8ep_seed20260921.yaml",
        "artifacts/experiments/20260922-stage2-r4-ac-tailmeanfloor035-8ep-seed20260921",
        "AC tail 50% mean-weight floor 0.35; all other R3 variables frozen",
    ),
    QueueItem(
        "R4-5",
        "configs/experiment/stage2_r4_ac_minmass4_8ep_seed20260921.yaml",
        "artifacts/experiments/20260922-stage2-r4-ac-minmass4-8ep-seed20260921",
        "AC per-class effective mass floor 4.0; all other R3 variables frozen",
    ),
]


def main() -> int:
    return run_serial_queue(
        root=ROOT,
        python=Path(sys.executable),
        logs=LOGS,
        queue_name="20260922-stage2-r4-ac",
        items=QUEUE,
    )


if __name__ == "__main__":
    raise SystemExit(main())
