"""Run the approved Stage2 Day-3 learning-rate candidates serially."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import QueueItem, run_serial_queue  # noqa: E402


ITEMS = [
    QueueItem(
        "L1-lr125",
        "configs/experiment/stage2_d3_lr125_8ep_seed20260921.yaml",
        "artifacts/experiments/20260923-stage2-d3-lr125-8ep-seed20260921",
        "last-4 AC with only visual learning rate changed to 1.25e-6",
    ),
    QueueItem(
        "L2-lr225",
        "configs/experiment/stage2_d3_lr225_8ep_seed20260921.yaml",
        "artifacts/experiments/20260923-stage2-d3-lr225-8ep-seed20260921",
        "last-4 AC with only visual learning rate changed to 2.25e-6",
    ),
]


def main() -> int:
    return run_serial_queue(
        root=ROOT,
        python=Path(sys.executable),
        logs=ROOT / "artifacts/queues/20260923-stage3-lr-serial",
        queue_name="20260923-stage3-lr-serial",
        items=ITEMS,
    )


if __name__ == "__main__":
    raise SystemExit(main())
