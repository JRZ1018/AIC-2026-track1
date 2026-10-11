"""Retry C5 fold-2 PEFT after the initial-parity config omission.

The 336 feature cache and matching H1 were completed and are reused unchanged.
Only this fresh PEFT output is launched under the repository-wide atomic lock.
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import QueueItem, run_serial_queue  # noqa: E402

DEADLINE = datetime(2026, 9, 25, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


def launch_until_deadline(command, cwd, output, on_start):
    if datetime.now(ZoneInfo("Asia/Shanghai")) >= DEADLINE:
        raise TimeoutError("C5 retry deadline passed before launch")
    child = subprocess.Popen(
        list(command), cwd=cwd, stdout=output, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    on_start(child.pid)
    while child.poll() is None:
        if datetime.now(ZoneInfo("Asia/Shanghai")) >= DEADLINE:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                raise RuntimeError("C5 child termination unconfirmed; retain lock for inspection")
            output.write("C5 stopped at 09:00; inspect complete last.pt before resuming\n")
            output.flush()
            return 124
        time.sleep(5)
    return child.returncode


def main() -> int:
    return run_serial_queue(
        root=ROOT,
        python=Path(sys.executable),
        logs=ROOT / "artifacts" / "queues" / "20260925-stage2-c5-fold2-336-retry",
        queue_name="20260925-stage2-c5-fold2-336-retry",
        items=[QueueItem(
            tag="C5-fold2-336-PEFT",
            config="configs/experiment/stage2_fold2_c5_336_last8_ac_8ep_seed20260921.yaml",
            output_dir="artifacts/experiments/20260925-stage2-fold2-c5-336-last8-ac-8ep-seed20260921",
            purpose="8-epoch last-8 AC from official CLIP and fresh fold-2 336 H1",
        )],
        launcher=launch_until_deadline,
    )


if __name__ == "__main__":
    raise SystemExit(main())
