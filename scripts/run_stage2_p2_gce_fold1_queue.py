"""Run the pre-registered 336/GCE fold-1 PEFT screening under the atomic lock."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aictrack1.engine.training_queue import TrainingLease  # noqa: E402

QUEUE = ROOT / "artifacts/queues/20260926-stage2-p2-gce-fold1"
CONFIG = ROOT / "configs/experiment/stage2_p2_gce_fold1_336_last8_ac_8ep_q07_seed20260921.yaml"
OUTPUT = ROOT / "artifacts/experiments/20260926-stage2-p2-gce-fold1-336-last8-ac-8ep-q07-seed20260921"
DEADLINE = datetime.now(ZoneInfo("Asia/Shanghai")) + timedelta(hours=12)


def main() -> int:
    if OUTPUT.exists() or QUEUE.exists():
        raise FileExistsError("P2 output or queue already exists; refuse to overwrite")
    lease = TrainingLease.acquire(
        ROOT / "artifacts/queues/training.lock.json",
        owner="20260926-stage2-p2-gce-fold1",
        config=str(CONFIG), output_dir=str(OUTPUT), command=tuple(sys.argv),
    )
    try:
        QUEUE.mkdir(parents=True, exist_ok=False)
        started = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        (QUEUE / "chain.log").write_text(
            f"[{started}] P2 GCE fold1 started; 336 train/validation only; no test access\n",
            encoding="utf-8",
        )
        with (QUEUE / "peft-gce-fold1.log").open("w", encoding="utf-8", newline="\n") as log:
            child = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts/train_peft.py"), "--config", str(CONFIG)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            lease.attach_trainer(child.pid)
            while child.poll() is None:
                if datetime.now(ZoneInfo("Asia/Shanghai")) >= DEADLINE:
                    child.terminate()
                    child.wait(timeout=30)
                    raise TimeoutError("approved 12-hour deadline stopped P2 GCE")
                time.sleep(10)
            if child.returncode != 0:
                raise RuntimeError(f"P2 GCE PEFT failed with exit code {child.returncode}")
        metrics = json.loads((OUTPUT / "metrics.json").read_text(encoding="utf-8"))
        if len(metrics.get("history", [])) != 8 or not (OUTPUT / "best.pt").is_file():
            raise RuntimeError("P2 GCE output incomplete")
        with (QUEUE / "chain.log").open("a", encoding="utf-8", newline="\n") as log:
            log.write(
                f"[{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}] complete best_top1={metrics.get('best_top1')}\n"
            )
    finally:
        lease.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
