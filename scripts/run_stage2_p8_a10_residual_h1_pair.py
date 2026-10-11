"""Run the approved P8 fold2/fold3 residual H1 screen under one atomic lock."""

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
        logs=ROOT / "artifacts" / "queues" / "20260928-p8-a10-residual-h1-pair",
        queue_name="20260928-p8-a10-residual-h1-pair",
        items=[
            QueueItem(
                "fold2-residual-H1",
                "configs/experiment/stage2_p8_fold2_residual_h1_336_seed20260921.yaml",
                "artifacts/experiments/20260928-stage2-p8-fold2-residual-h1-336-seed20260921",
                "P8 fold2 residual nonlinear head on frozen official CLIP features",
                trainer_entrypoint="scripts/train_frozen_head.py",
            ),
            QueueItem(
                "fold3-residual-H1",
                "configs/experiment/stage2_p8_fold3_residual_h1_336_seed20260921.yaml",
                "artifacts/experiments/20260928-stage2-p8-fold3-residual-h1-336-seed20260921",
                "P8 fold3 residual nonlinear head on frozen official CLIP features",
                trainer_entrypoint="scripts/train_frozen_head.py",
            ),
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
