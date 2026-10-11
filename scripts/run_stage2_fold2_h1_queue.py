"""Train the Stage 2 seeded-random fold-2 H1 under the atomic training lock."""

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
        logs=ROOT / "artifacts" / "queues" / "20260924-stage2-fold2-h1",
        queue_name="20260924-stage2-fold2-h1",
        items=[
            QueueItem(
                "fold2-H1",
                "configs/experiment/stage2_fold2_h1_linear_seed20260921.yaml",
                "artifacts/experiments/20260924-stage2-fold2-h1-linear-seed20260921",
                "fold-2 fresh Stage 2 H1 from official CLIP features",
                trainer_entrypoint="scripts/train_frozen_head.py",
            ),
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
