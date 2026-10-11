"""Run approved D2/D3 C2-feature heads under the repository training lock."""

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
        logs=ROOT / "artifacts" / "queues" / "20260923-d2-d3-c2-heads",
        queue_name="20260923-d2-d3-c2-heads",
        items=[
            QueueItem(
                "D2-invsqrt-p05",
                "configs/experiment/stage2_d2_c2_features_invsqrt_p05_seed20260921.yaml",
                "artifacts/experiments/20260923-stage2-d2-c2-features-invsqrt-p05-seed20260921",
                "one-variable inverse-sqrt class-weighted C2 linear head",
                trainer_entrypoint="scripts/train_frozen_head.py",
            ),
            QueueItem(
                "D3-cosine",
                "configs/experiment/stage2_d3_c2_features_cosine_seed20260921.yaml",
                "artifacts/experiments/20260923-stage2-d3-c2-features-cosine-seed20260921",
                "one-variable C2 cosine head",
                trainer_entrypoint="scripts/train_frozen_head.py",
            ),
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
