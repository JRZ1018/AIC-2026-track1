"""Finalize the completed fold2 pair from exact epoch-8 checkpoints.

This queue performs metadata-only resumes after the archive provenance fallback
has been verified.  Both checkpoints already contain epoch 8 of 8, so the
trainer loop is skipped and only selected checkpoints plus run-local metadata
are regenerated under the normal atomic training lease.
"""

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import QueueItem, run_serial_queue


def main() -> int:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("the A10 fold2 finalizer is approved for Linux only")
    if shutil.disk_usage(ROOT).free < 4 * 1024**3:
        raise RuntimeError("the A10 fold2 finalizer requires at least 4 GiB free")
    commit = os.environ.get("AICTRACK1_GIT_COMMIT", "").strip().lower()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("AICTRACK1_GIT_COMMIT must be the verified 40-character source commit")
    items = [
        QueueItem(
            tag="gce_q07_control_finalize",
            config="configs/experiment/stage2_p5_sce_fold2_a10_control.yaml",
            output_dir="artifacts/experiments/20260928-stage2-p5-fold2-a10-gce-q07-control",
            purpose="metadata-only resume from completed epoch-8 GCE checkpoint",
            resume_checkpoint=(
                "artifacts/experiments/20260928-stage2-p5-fold2-a10-gce-q07-control/last.pt"
            ),
        ),
        QueueItem(
            tag="sce_candidate_finalize",
            config="configs/experiment/stage2_p5_sce_fold2_a10_candidate.yaml",
            output_dir="artifacts/experiments/20260928-stage2-p5-fold2-a10-sce-candidate",
            purpose="metadata-only resume from completed epoch-8 SCE checkpoint",
            resume_checkpoint=(
                "artifacts/experiments/20260928-stage2-p5-fold2-a10-sce-candidate/last.pt"
            ),
        ),
    ]
    return run_serial_queue(
        root=ROOT,
        python=Path(sys.executable),
        logs=ROOT / "artifacts/queues/20260928-p5-a10-fold2-finalize",
        queue_name="20260928-p5-a10-fold2-finalize",
        items=items,
    )


if __name__ == "__main__":
    raise SystemExit(main())
