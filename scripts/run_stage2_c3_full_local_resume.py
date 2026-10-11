"""Resume the single C3 full-data PEFT run after an interrupted queue."""

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
        logs=ROOT / "artifacts" / "queues" / "20260924-stage2-c3-full-local-resume",
        queue_name="20260924-stage2-c3-full-local-resume",
        items=[
            QueueItem(
                "C3-full-last8-resume",
                "configs/experiment/stage2_c3_peft_ac_last8_full_14ep_seed20260921.yaml",
                "artifacts/experiments/20260924-stage2-c3-peft-ac-last8-full-14ep-seed20260921",
                "resume same full-data C3 model from complete epoch-1 last.pt",
                resume_checkpoint="artifacts/experiments/20260924-stage2-c3-peft-ac-last8-full-14ep-seed20260921/last.pt",
            ),
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
