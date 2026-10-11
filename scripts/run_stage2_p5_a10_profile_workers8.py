"""Two-epoch A10 workers=8 resource profile; train-derived fold only."""

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import QueueItem, run_serial_queue


def main() -> int:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("the A10 profile is preregistered for Linux only")
    if shutil.disk_usage(ROOT).free < 4 * 1024**3:
        raise RuntimeError("the A10 profile requires at least 4 GiB free")
    required_ld_paths = (
        "/usr/local/cuda-12.8/extras/CUPTI/lib64",
        "/usr/local/cuda-12.8/targets/x86_64-linux/lib",
        "/usr/local/cuda-12.8/lib64",
    )
    current = os.environ.get("LD_LIBRARY_PATH", "")
    missing = [path for path in required_ld_paths if path not in current.split(":")]
    if missing:
        raise RuntimeError(f"missing required CUDA runtime paths: {missing}")
    item = QueueItem(
        tag="gce_q07_workers8_profile",
        config="configs/experiment/stage2_p5_fold2_a10_gce_profile_workers8_2ep.yaml",
        output_dir="artifacts/experiments/20260928-stage2-p5-fold2-a10-gce-q07-profile-workers8-2ep",
        purpose="A10 workers=8 two-epoch resource profile on fold2 train only",
    )
    return run_serial_queue(
        root=ROOT,
        python=Path(sys.executable),
        logs=ROOT / "artifacts/queues/20260928-p5-a10-profile-workers8",
        queue_name="20260928-p5-a10-profile-workers8",
        items=[item],
    )


if __name__ == "__main__":
    raise SystemExit(main())
