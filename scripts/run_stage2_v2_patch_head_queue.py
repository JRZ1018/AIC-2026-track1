"""Run the V2 patch-readout head after the approved train-only cache exists."""

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
        logs=ROOT / "artifacts" / "queues" / "20260924-v2-patch-head",
        queue_name="20260924-v2-patch-head",
        items=[
            QueueItem(
                "V2-patch-head",
                "configs/experiment/stage2_v2_c2_cls_patchmean_linear_head_seed20260921.yaml",
                "artifacts/experiments/20260924-stage2-v2-c2-cls-patchmean-linear-head-seed20260921",
                "same C2 features/seed/schedule as D1; append final patch mean",
                trainer_entrypoint="scripts/train_frozen_head.py",
            )
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
