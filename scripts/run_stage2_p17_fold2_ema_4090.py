"""Bounded, test-free P17 fold2 EMA screen on the approved RTX4090."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aictrack1.engine.training_queue import (  # noqa: E402
    QueueItem, probe_running_trainers, read_training_lock, run_serial_queue,
)

NAME = "20261001-p17-fold2-ema0999-4090-w16-v4"
ORIGINAL = Path("/root/autodl-tmp/AICtrack1")
EXPECTED_ROOT = Path("/root/autodl-tmp/AICtrack1_p17_ema0999_20261001_w16_v4")
QUEUE = ROOT / "artifacts/queues" / NAME
LOCK = ORIGINAL / "artifacts/queues/training.lock.json"
CONFIG = "configs/experiment/stage2_p17_fold2_sce_ac_ema0999_8ep_4090.yaml"
OUTPUT = "artifacts/experiments/20261001-stage2-p17-fold2-sce-ac-ema0999-8ep-4090-w16-v4"
HARD_SECONDS = 3 * 3600 + 30 * 60
MAX_ADDED = 4 * 1024**3
MIN_START_FREE = 9 * 1024**3
MIN_FREE = 5 * 1024**3


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if sys.platform != "linux" or ROOT != EXPECTED_ROOT:
        raise RuntimeError("P17 requires the exact isolated Linux workspace")
    if (ROOT / "DataSetStage2/test").exists() or QUEUE.exists() or (ROOT / OUTPUT).exists():
        raise RuntimeError("P17 test, queue or output already exists")
    if read_training_lock(LOCK) is not None or probe_running_trainers():
        raise RuntimeError("shared lock or another trainer active")
    if not torch.cuda.is_available() or "RTX 4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("approved RTX4090 unavailable")
    disk_free = shutil.disk_usage(ROOT).free
    gpu_free, _ = torch.cuda.mem_get_info(0)
    if disk_free < MIN_START_FREE or gpu_free < 20 * 1024**3:
        raise RuntimeError(f"P17 resource guard failed: gpu={gpu_free}, disk={disk_free}")
    expected = json.loads((ROOT / "p17_input_sha256.json").read_text(encoding="utf-8"))
    drift = {name: digest for name, digest in expected.items() if not (ROOT / name).is_file() or sha(ROOT / name) != digest}
    if drift:
        raise RuntimeError(f"P17 frozen source/input SHA mismatch: {sorted(drift)}")
    started = datetime.now(ZoneInfo("Asia/Shanghai"))
    deadline = started + timedelta(seconds=HARD_SECONDS)
    QUEUE.mkdir(parents=True, exist_ok=False)
    (QUEUE / "source_receipt.json").write_text(json.dumps({
        "scope": "Stage2 train-only fold2 EMA short screen",
        "started_at": started.isoformat(), "deadline": deadline.isoformat(),
        "queue_pid": os.getpid(), "gpu": torch.cuda.get_device_name(0),
        "disk_free_start": disk_free, "gpu_free_start": gpu_free,
        "max_added_bytes": MAX_ADDED, "source_sha256": expected,
    }, indent=2) + "\n", encoding="utf-8")

    def launch(command, cwd, output, on_start) -> int:
        if any(sha(ROOT / name) != digest for name, digest in expected.items()):
            raise RuntimeError("P17 frozen input changed before launch")
        child = subprocess.Popen(list(command), cwd=cwd, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            on_start(child.pid)
        except Exception:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=30)
            raise
        while True:
            try:
                return child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                free = shutil.disk_usage(ROOT).free
                if time.time() >= deadline.timestamp() or free < MIN_FREE or disk_free - free > MAX_ADDED:
                    output.write(f"P17 hard/resource cutoff: free={free}, added={disk_free-free}\n")
                    output.flush()
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=30)
                    return 124

    code = run_serial_queue(
        root=ROOT, python=Path(sys.executable), logs=QUEUE, queue_name=NAME,
        items=[QueueItem("ema_fold2_8ep", CONFIG, OUTPUT, "P17 SCE+AC EMA0.999 fold2 short screen")],
        launcher=launch, lock_path=LOCK,
    )
    (QUEUE / "queue_result.json").write_text(json.dumps({
        "exit_code": code, "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
