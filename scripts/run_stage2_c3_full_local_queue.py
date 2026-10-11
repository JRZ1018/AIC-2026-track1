"""Train fresh Stage 2 full-data H1 and one C3 model under the atomic lock."""

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
        logs=ROOT / "artifacts" / "queues" / "20260924-stage2-c3-full-local",
        queue_name="20260924-stage2-c3-full-local",
        items=[
            QueueItem(
                "C3-full-H1",
                "configs/experiment/stage2_c3_h1_full_seed20260921.yaml",
                "artifacts/experiments/20260924-stage2-c3-h1-full-seed20260921",
                "fresh linear H1 on all Stage 2 train CLIP features",
                trainer_entrypoint="scripts/train_frozen_head.py",
            ),
            QueueItem(
                "C3-full-last8",
                "configs/experiment/stage2_c3_peft_ac_last8_full_14ep_seed20260921.yaml",
                "artifacts/experiments/20260924-stage2-c3-peft-ac-last8-full-14ep-seed20260921",
                "single-model Stage 2 full-train last-8 AC for 14 fixed epochs",
            ),
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
