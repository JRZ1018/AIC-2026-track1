"""Matched fold3 GCE/SCE confirmation in one controlled A10 batch."""

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import QueueItem, run_parallel_queue


def main() -> int:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("the A10 fold3 pair is preregistered for Linux only")
    if shutil.disk_usage(ROOT).free < 8 * 1024**3:
        raise RuntimeError("the controlled A10 pair requires at least 8 GiB free")
    required_ld_paths = (
        "/usr/local/cuda-12.8/extras/CUPTI/lib64",
        "/usr/local/cuda-12.8/targets/x86_64-linux/lib",
        "/usr/local/cuda-12.8/lib64",
    )
    current = os.environ.get("LD_LIBRARY_PATH", "")
    missing = [path for path in required_ld_paths if path not in current.split(":")]
    if missing:
        raise RuntimeError(f"missing required CUDA runtime paths: {missing}")
    items = [
        QueueItem(
            tag="gce_q07_control",
            config="configs/experiment/stage2_p5_sce_fold3_a10_control.yaml",
            output_dir="artifacts/experiments/20260928-stage2-p5-fold3-a10-gce-q07-control",
            purpose="fold3 matched GCE q=.7 control, batch64 workers4",
        ),
        QueueItem(
            tag="sce_candidate",
            config="configs/experiment/stage2_p5_sce_fold3_a10_candidate.yaml",
            output_dir="artifacts/experiments/20260928-stage2-p5-fold3-a10-sce-candidate",
            purpose="fold3 fixed SCE(.1,1,-4) candidate, batch64 workers4",
        ),
    ]
    return run_parallel_queue(
        root=ROOT,
        python=Path(sys.executable),
        logs=ROOT / "artifacts/queues/20260928-p5-a10-fold3-parallel",
        queue_name="20260928-p5-a10-fold3-parallel",
        items=items,
        max_concurrent_jobs=2,
    )


if __name__ == "__main__":
    raise SystemExit(main())
