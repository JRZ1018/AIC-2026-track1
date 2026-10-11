"""Run the preregistered Stage 2 fold-2 last-12 capacity screen."""

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
        logs=ROOT / "artifacts" / "queues" / "20260924-stage2-c4-fold2-local",
        queue_name="20260924-stage2-c4-fold2-local",
        items=[
            QueueItem(
                "C4-fold2-last12",
                "configs/experiment/stage2_fold2_c4_last12_ac_8ep_seed20260921.yaml",
                "artifacts/experiments/20260924-stage2-fold2-c4-last12-ac-8ep-seed20260921",
                "single-variable last-8 to last-12 visual capacity screen on fold 2",
            ),
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
