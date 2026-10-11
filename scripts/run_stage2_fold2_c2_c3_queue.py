"""Run paired Stage 2 fold-2 C2/C3 visual screens under the atomic lock."""

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
        logs=ROOT / "artifacts" / "queues" / "20260924-stage2-fold2-c2-c3",
        queue_name="20260924-stage2-fold2-c2-c3",
        items=[
            QueueItem(
                "fold2-C2-last6",
                "configs/experiment/stage2_fold2_c2_last6_ac_8ep_seed20260921.yaml",
                "artifacts/experiments/20260924-stage2-fold2-c2-last6-ac-8ep-seed20260921",
                "fold-2 frozen-head last-6 control",
            ),
            QueueItem(
                "fold2-C3-last8",
                "configs/experiment/stage2_fold2_c3_last8_ac_8ep_seed20260921.yaml",
                "artifacts/experiments/20260924-stage2-fold2-c3-last8-ac-8ep-seed20260921",
                "fold-2 frozen-head last-8 candidate",
            ),
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
