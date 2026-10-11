"""Gate-dependent full Stage 2 matched-336 H1 and single PEFT model."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import QueueItem, run_serial_queue  # noqa: E402
from run_stage2_fold3_peft_pair_queue import launch_until_deadline  # noqa: E402


def main() -> int:
    return run_serial_queue(
        root=ROOT,
        python=Path(sys.executable),
        logs=ROOT / "artifacts/queues/20260925-stage2-c6-336-full",
        queue_name="20260925-stage2-c6-336-full",
        items=[
            QueueItem(
                tag="c6-336-full-h1",
                config="configs/experiment/stage2_c6_h1_336_full_seed20260921.yaml",
                output_dir="artifacts/experiments/20260925-stage2-c6-h1-336-full-seed20260921",
                purpose="new full-train H1 from official CLIP Stage2-only 336 features",
                trainer_entrypoint="scripts/train_frozen_head.py",
            ),
            QueueItem(
                tag="c6-336-full-peft",
                config="configs/experiment/stage2_c6_peft_ac_last8_336_full_14ep_seed20260921.yaml",
                output_dir="artifacts/experiments/20260925-stage2-c6-peft-ac-last8-336-full-14ep-seed20260921",
                purpose="single 14-epoch 336 AC model from official CLIP and new full H1",
            ),
        ],
        launcher=launch_until_deadline,
    )


if __name__ == "__main__":
    raise SystemExit(main())
