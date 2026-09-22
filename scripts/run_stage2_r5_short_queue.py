"""Run the preregistered two-seed Stage2 R5 short confirmation serially."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import QueueItem, run_serial_queue  # noqa: E402


LOGS = ROOT / "artifacts" / "queues" / "20260922-stage2-r5-short"
QUEUE = [
    QueueItem(
        "R5-seed20260922",
        "configs/experiment/stage2_r5_peft_ac_last4_8ep_seed20260922.yaml",
        "artifacts/experiments/20260922-stage2-r5-peft-ac-last4-8ep-seed20260922",
        "R3 AC recipe short confirmation, seed 20260922",
    ),
    QueueItem(
        "R5-seed20260923",
        "configs/experiment/stage2_r5_peft_ac_last4_8ep_seed20260923.yaml",
        "artifacts/experiments/20260923-stage2-r5-peft-ac-last4-8ep-seed20260923",
        "R3 AC recipe short confirmation, seed 20260923",
    ),
]


def main() -> int:
    return run_serial_queue(
        root=ROOT,
        python=Path(sys.executable),
        logs=LOGS,
        queue_name="20260922-stage2-r5-short",
        items=QUEUE,
    )


if __name__ == "__main__":
    raise SystemExit(main())
