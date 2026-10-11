"""Run the approved single-variable V1 joint-head experiment under the lock."""

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
        logs=ROOT / "artifacts" / "queues" / "20260923-v1-joint-head-retry2",
        queue_name="20260923-v1-joint-head-retry2",
        items=[
            QueueItem(
                "V1-joint-head",
                "configs/experiment/stage2_v1_joint_head_last6_ac_8ep_seed20260921.yaml",
                "artifacts/experiments/20260923-stage2-v1-joint-head-last6-ac-8ep-seed20260921",
                "last-6 AC versus frozen-head C2; only head lr 0 to 1e-5",
            )
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
