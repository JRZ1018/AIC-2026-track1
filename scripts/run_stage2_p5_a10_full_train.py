"""Train the new full H1 and one full 14-epoch SCE model on A10."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import QueueItem, run_serial_queue  # noqa: E402

DEADLINE = datetime(2026, 9, 28, 14, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


def launch_until_deadline(command, cwd, log, attach):
    if datetime.now(ZoneInfo("Asia/Shanghai")) >= DEADLINE:
        raise TimeoutError("P5 full training cutoff reached")
    child = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=(os.name != "nt"),
    )
    attach(child.pid)
    while child.poll() is None:
        if datetime.now(ZoneInfo("Asia/Shanghai")) >= DEADLINE:
            if os.name == "nt":
                child.terminate()
            else:
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    child.kill()
                else:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=30)
            return child.returncode
        time.sleep(5)
    return child.returncode


def main() -> int:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("the P5 A10 full train route is Linux-only")
    if shutil.disk_usage(ROOT).free < 8 * 1024**3:
        raise RuntimeError("the P5 A10 full train route requires 8 GiB free")
    if not (ROOT / "artifacts/features/stage2/train_official336_p5_20260928/complete.json").is_file():
        raise FileNotFoundError("validated P5 full feature cache is absent")
    return run_serial_queue(
        root=ROOT,
        python=Path(sys.executable),
        logs=ROOT / "artifacts/queues/20260928-p5-a10-full-train",
        queue_name="20260928-p5-a10-full-train",
        items=[
            QueueItem(
                tag="p5-sce-full-h1",
                config="configs/experiment/stage2_p5_sce_h1_336_full_a10.yaml",
                output_dir="artifacts/experiments/20260928-stage2-p5-sce-h1-336-full-seed20260921",
                purpose="new full H1 from official OpenAI CLIP Stage2-only 336 features",
                trainer_entrypoint="scripts/train_frozen_head.py",
                resume_checkpoint=(
                    "artifacts/experiments/"
                    "20260928-stage2-p5-sce-h1-336-full-seed20260921/last.pt"
                ),
            ),
            QueueItem(
                tag="p5-sce-full-peft",
                config="configs/experiment/stage2_p5_sce_peft_336_full_14ep_a10.yaml",
                output_dir="artifacts/experiments/20260928-stage2-p5-sce-336-full-14ep-seed20260921",
                purpose="single full-train 14-epoch SCE model from new full H1",
            ),
        ],
        launcher=launch_until_deadline,
    )


if __name__ == "__main__":
    raise SystemExit(main())
