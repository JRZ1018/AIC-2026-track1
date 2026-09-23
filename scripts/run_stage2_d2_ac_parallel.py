"""Run the approved Stage2 Day-2 AC single-variable candidates in parallel."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import QueueItem, run_parallel_queue  # noqa: E402


ITEMS = [
    QueueItem(
        "A1-minweight005",
        "configs/experiment/stage2_d2_ac_minweight005_8ep_seed20260921.yaml",
        "artifacts/experiments/20260923-stage2-d2-ac-minweight005-8ep-seed20260921",
        "AC baseline with only min_weight changed from 0.02 to 0.05",
    ),
    QueueItem(
        "A2-magmix050",
        "configs/experiment/stage2_d2_ac_magmix050_8ep_seed20260921.yaml",
        "artifacts/experiments/20260923-stage2-d2-ac-magmix050-8ep-seed20260921",
        "AC baseline with only magnitude_mix changed from 0.75 to 0.50",
    ),
]


def main() -> int:
    return run_parallel_queue(
        root=ROOT,
        python=Path(sys.executable),
        logs=ROOT / "artifacts/queues/20260923-stage2-d2-ac-parallel",
        queue_name="20260923-stage2-d2-ac-parallel",
        items=ITEMS,
        max_concurrent_jobs=2,
    )


if __name__ == "__main__":
    raise SystemExit(main())
