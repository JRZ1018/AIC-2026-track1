"""Run the preregistered C3 last-8 fixed-fold screen under the training lock."""

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
        logs=ROOT / "artifacts" / "queues" / "20260924-c3-last8-screen",
        queue_name="20260924-c3-last8-screen",
        items=[
            QueueItem(
                "C3-seed20260921",
                "configs/experiment/stage2_c3_last8_ac_8ep_seed20260921.yaml",
                "artifacts/experiments/20260924-stage2-c3-last8-ac-8ep-seed20260921",
                "last-8 visual capacity fixed-fold screen, seed 20260921",
            ),
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
