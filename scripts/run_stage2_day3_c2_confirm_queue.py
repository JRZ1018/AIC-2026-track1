"""Run the preregistered C2 last-6 seed confirmation under the training lock."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import QueueItem, run_serial_queue  # noqa: E402


def main() -> int:
    return run_serial_queue(
        root=ROOT,
        python=Path(sys.executable),
        logs=ROOT / "artifacts" / "queues" / "20260923-day3-c2-confirm",
        queue_name="20260923-day3-c2-confirm",
        items=[
            QueueItem(
                "C2-seed20260922",
                "configs/experiment/stage2_day3_c2_last6_ac_8ep_seed20260922.yaml",
                "artifacts/experiments/20260923-stage2-day3-c2-last6-ac-8ep-seed20260922",
                "last-6 visual capacity confirmation, seed 20260922",
            ),
            QueueItem(
                "C2-seed20260923",
                "configs/experiment/stage2_day3_c2_last6_ac_8ep_seed20260923.yaml",
                "artifacts/experiments/20260923-stage2-day3-c2-last6-ac-8ep-seed20260923",
                "last-6 visual capacity confirmation, seed 20260923",
            ),
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
