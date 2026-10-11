"""Resume the fold-1 320 screening at PEFT after H1 completion."""

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

QUEUE = ROOT / "artifacts/queues/20260926-stage2-c8-320-fold1-peft-resume"
CONFIG = ROOT / "configs/experiment/stage2_fold1_c8_320_last8_ac_8ep_seed20260921.yaml"
FEATURES = ROOT / "artifacts/features/stage2/train_openai_vit_b32_fold1_320_20260926"
H1 = ROOT / "artifacts/experiments/20260926-stage2-fold1-c8-h1-320-seed20260921"
PEFT = ROOT / "artifacts/experiments/20260926-stage2-fold1-c8-320-last8-ac-8ep-seed20260921"
DEADLINE = datetime.now(ZoneInfo("Asia/Shanghai")) + timedelta(hours=12)


def main() -> int:
    for path in (CONFIG, FEATURES / "complete.json", H1 / "best.pt", H1 / "metrics.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (QUEUE, PEFT):
        if path.exists():
            raise FileExistsError(path)
    lease = TrainingLease.acquire(
        ROOT / "artifacts/queues/training.lock.json",
        owner="20260926-stage2-c8-320-fold1-peft-resume",
        config=str(CONFIG), output_dir=str(PEFT), command=tuple(sys.argv),
    )
    try:
        QUEUE.mkdir(parents=True, exist_ok=False)
        with (QUEUE / "chain.log").open("w", encoding="utf-8", newline="\n") as log:
            log.write(f"[{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}] PEFT resume started; feature/H1 verified; no test access\n")
        with (QUEUE / "peft-320.log").open("w", encoding="utf-8", newline="\n") as handle:
            child = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts/train_peft.py"), "--config", str(CONFIG)],
                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            lease.attach_trainer(child.pid)
            while child.poll() is None:
                if datetime.now(ZoneInfo("Asia/Shanghai")) >= DEADLINE:
                    child.terminate()
                    child.wait(timeout=30)
                    raise TimeoutError("approved 12-hour deadline stopped PEFT")
                time.sleep(10)
            if child.returncode != 0:
                raise RuntimeError(f"PEFT failed with exit code {child.returncode}")
        metrics = json.loads((PEFT / "metrics.json").read_text(encoding="utf-8"))
        if len(metrics["history"]) != 8 or not (PEFT / "best.pt").is_file():
            raise RuntimeError("320 PEFT output incomplete")
        with (QUEUE / "chain.log").open("a", encoding="utf-8", newline="\n") as log:
            log.write(f"[{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}] PEFT complete best_top1={metrics.get('best_top1')}\n")
    finally:
        lease.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
