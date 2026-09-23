"""Run the frozen single-model Stage2 C2 full retrain under the training lock."""

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
        logs=ROOT / "artifacts" / "queues" / "20260923-stage2-c2-final",
        queue_name="20260923-stage2-c2-final",
        items=[
            QueueItem(
                "C2-final",
                "configs/experiment/stage2_c2_peft_ac_last6_full_14ep_seed20260921.yaml",
                "artifacts/experiments/20260923-stage2-c2-peft-ac-last6-full-14ep-seed20260921",
                "single-model Stage2 full-train last-6 AC, frozen 14 epochs",
            )
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
