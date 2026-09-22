"""Run the two unlocked Stage2 R4 CE long-tail screens serially.

The queue is intentionally limited to R4-2 and R4-3. AC protection candidates remain
disabled until both CE screens are measured against the preregistered epoch-8 gates.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import QueueItem, run_serial_queue  # noqa: E402


LOGS = ROOT / "artifacts" / "queues" / "20260922-stage2-r4-ce"
QUEUE = [
    QueueItem(
        "R4-2",
        "configs/experiment/stage2_r4_ce_classbalanced_p05_8ep_seed20260921.yaml",
        "artifacts/experiments/20260922-stage2-r4-ce-classbalanced-p05-8ep-seed20260921",
        "mild count^-0.5 class-balanced sampling; all other R2 variables frozen",
    ),
    QueueItem(
        "R4-3",
        "configs/experiment/stage2_r4_ce_logitadj_t05_8ep_seed20260921.yaml",
        "artifacts/experiments/20260922-stage2-r4-ce-logitadj-t05-8ep-seed20260921",
        "train-only tau=0.5 logit adjustment; natural sampling and all other variables frozen",
    ),
]


def main() -> int:
    return run_serial_queue(
        root=ROOT,
        python=Path(sys.executable),
        logs=LOGS,
        queue_name="20260922-stage2-r4-ce",
        items=QUEUE,
    )


if __name__ == "__main__":
    raise SystemExit(main())
