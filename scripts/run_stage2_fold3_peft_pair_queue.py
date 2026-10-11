"""Train the preregistered Stage 2 fold-3 224/336 PEFT pair serially."""

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

DEADLINE = datetime(2026, 9, 25, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


def launch_until_deadline(command, cwd, output, on_start):
    if datetime.now(ZoneInfo("Asia/Shanghai")) >= DEADLINE:
        raise TimeoutError("approved resource deadline reached before fold-3 launch")
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
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("fold-3 trainer termination unconfirmed; retain lock") from exc
            output.write("Stopped at approved 20:00 deadline; inspect complete last.pt before resume\n")
            output.flush()
            return 124
        time.sleep(5)
    return child.returncode


def main() -> int:
    return run_serial_queue(
        root=ROOT,
        python=Path(sys.executable),
        logs=ROOT / "artifacts/queues/20260925-stage2-fold3-peft-pair",
        queue_name="20260925-stage2-fold3-peft-pair",
        items=[
            QueueItem(
                tag="fold3-c3-224",
                config="configs/experiment/stage2_fold3_c3_last8_ac_8ep_seed20260921.yaml",
                output_dir="artifacts/experiments/20260925-stage2-fold3-c3-last8-ac-8ep-seed20260921",
                purpose="fold-specific 224 C3 baseline from official CLIP and new H1",
            ),
            QueueItem(
                tag="fold3-c5-336",
                config="configs/experiment/stage2_fold3_c5_336_last8_ac_8ep_seed20260921.yaml",
                output_dir="artifacts/experiments/20260925-stage2-fold3-c5-336-last8-ac-8ep-seed20260921",
                purpose="matched-336 C5 candidate from official CLIP and new H1",
            ),
        ],
        launcher=launch_until_deadline,
    )


if __name__ == "__main__":
    raise SystemExit(main())
