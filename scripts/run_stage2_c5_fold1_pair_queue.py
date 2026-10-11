"""Run the old fixed-fold 224/C3 and matched-336/C5 pair serially."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import QueueItem, run_serial_queue  # noqa: E402
from run_stage2_c5_fold2_336_retry import launch_until_deadline  # noqa: E402


def main() -> int:
    return run_serial_queue(
        root=ROOT,
        python=Path(sys.executable),
        logs=ROOT / "artifacts" / "queues" / "20260925-stage2-c5-fold1-pair",
        queue_name="20260925-stage2-c5-fold1-pair",
        items=[
            QueueItem(
                tag="C3-fold1-224-local",
                config="configs/experiment/stage2_fold1_c3_last8_ac_8ep_local_confirm_seed20260921.yaml",
                output_dir="artifacts/experiments/20260925-stage2-fold1-c3-last8-ac-8ep-local-confirm-seed20260921",
                purpose="same-machine 224 C3 reference; original fold-1 H1, 8 epochs",
            ),
            QueueItem(
                tag="C5-fold1-336-local",
                config="configs/experiment/stage2_fold1_c5_336_last8_ac_8ep_seed20260921.yaml",
                output_dir="artifacts/experiments/20260925-stage2-fold1-c5-336-last8-ac-8ep-seed20260921",
                purpose="matched-336 C5 independent fold; new fold-1 H1, 8 epochs",
            ),
        ],
        launcher=launch_until_deadline,
    )


if __name__ == "__main__":
    raise SystemExit(main())
