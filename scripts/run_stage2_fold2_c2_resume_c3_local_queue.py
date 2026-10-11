"""Resume local fold-2 C2 from its verified epoch-7 checkpoint, then run C3."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import QueueItem, run_serial_queue  # noqa: E402


def main() -> int:
    c2_output = "artifacts/experiments/20260924-stage2-fold2-c2-last6-ac-8ep-seed20260921"
    return run_serial_queue(
        root=ROOT,
        python=Path(sys.executable),
        logs=ROOT / "artifacts" / "queues" / "20260924-stage2-fold2-c2-resume-c3-local",
        queue_name="20260924-stage2-fold2-c2-resume-c3-local",
        items=[
            QueueItem(
                "fold2-C2-last6-resume",
                "configs/experiment/stage2_fold2_c2_last6_ac_8ep_seed20260921.yaml",
                c2_output,
                "resume fold-2 C2 only from complete epoch-7 last.pt",
                resume_checkpoint=f"{c2_output}/last.pt",
            ),
            QueueItem(
                "fold2-C3-last8",
                "configs/experiment/stage2_fold2_c3_last8_ac_8ep_seed20260921.yaml",
                "artifacts/experiments/20260924-stage2-fold2-c3-last8-ac-8ep-seed20260921",
                "fold-2 frozen-head last-8 candidate from official CLIP",
            ),
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
