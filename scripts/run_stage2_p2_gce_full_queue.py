"""Run the gate-approved full-train 336/GCE PEFT model under the atomic lock."""

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

QUEUE = ROOT / "artifacts/queues/20260926-stage2-p2-gce-full"
CONFIG = ROOT / "configs/experiment/stage2_p2_gce_full_336_last8_ac_14ep_q07_seed20260921.yaml"
HEAD = ROOT / "artifacts/experiments/20260925-stage2-c6-h1-336-full-seed20260921/final.pt"
OUTPUT = ROOT / "artifacts/experiments/20260926-stage2-p2-gce-full-336-last8-ac-14ep-q07-seed20260921"
DEADLINE = datetime.now(ZoneInfo("Asia/Shanghai")) + timedelta(hours=12)


def main() -> int:
    if not HEAD.is_file():
        raise FileNotFoundError(HEAD)
    if OUTPUT.exists() or QUEUE.exists():
        raise FileExistsError("P2 full output or queue already exists; refuse to overwrite")
    lease = TrainingLease.acquire(
        ROOT / "artifacts/queues/training.lock.json",
        owner="20260926-stage2-p2-gce-full",
        config=str(CONFIG), output_dir=str(OUTPUT), command=tuple(sys.argv),
    )
    try:
        QUEUE.mkdir(parents=True, exist_ok=False)
        with (QUEUE / "chain.log").open("w", encoding="utf-8", newline="\n") as log:
            log.write(
                f"[{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}] P2 GCE full train started; 148695 train only; no test access\n"
            )
        with (QUEUE / "peft-gce-full.log").open("w", encoding="utf-8", newline="\n") as log:
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
                    raise TimeoutError("approved 12-hour deadline stopped P2 full GCE")
                time.sleep(10)
            if child.returncode != 0:
                raise RuntimeError(f"P2 full GCE PEFT failed with exit code {child.returncode}")
        metrics = json.loads((OUTPUT / "metrics.json").read_text(encoding="utf-8"))
        if len(metrics.get("history", [])) != 14 or not (OUTPUT / "final.pt").is_file():
            raise RuntimeError("P2 full GCE output incomplete")
        with (QUEUE / "chain.log").open("a", encoding="utf-8", newline="\n") as log:
            log.write(
                f"[{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}] complete best_top1={metrics.get('best_top1')}\n"
            )
    finally:
        lease.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
