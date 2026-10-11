"""Bounded train-only P14 fold2 diagnostic on the approved isolated server checkout."""

from __future__ import annotations

import hashlib
import json
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
from aictrack1.engine.training_queue import TrainingLease, probe_running_trainers, read_training_lock  # noqa: E402

ORIGINAL = Path("/root/autodl-tmp/AICtrack1")
SHARED_LOCK = ORIGINAL / "artifacts/queues/training.lock.json"
RUN = "20260930-stage2-p14-fold2-sce-ac-anchor01-8ep-4090"
CONFIG = "configs/experiment/stage2_p14_fold2_sce_ac_anchor01_8ep_4090.yaml"
BEST = ROOT / "artifacts/experiments" / RUN / "best.pt"
EXPECTED_BEST_SHA = "dc0d2d2b3e8ebe16411f0c9a594c6420b268d11a0dabfd7c30a3f56a66a94d45"
OUTPUT = ROOT / "artifacts/diagnostics/20260930-p14-fold2-anchor01-fourview384"
LOG = ROOT / "artifacts/queues/20260930-p14-fold2-anchor01-diagnostic.log"
RESULT = ROOT / "artifacts/queues/20260930-p14-fold2-anchor01-diagnostic-result.json"
LIMIT_SECONDS = 40 * 60
MAX_ADDED_BYTES = 1024**3
MIN_FREE_BYTES = 5 * 1024**3


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if sys.platform != "linux" or ROOT != Path("/root/autodl-tmp/AICtrack1_p14_anchor01"):
        raise RuntimeError("exact approved isolated server checkout required")
    if (ROOT / "DataSetStage2/test").exists():
        raise RuntimeError("test directory is forbidden")
    if OUTPUT.exists() or LOG.exists() or RESULT.exists():
        raise FileExistsError("diagnostic already started or produced output")
    queue = json.loads((ROOT / "artifacts/queues/20260930-p14-fold2-4090-anchor01-screen/queue_result.json").read_text())
    if queue.get("exit_code") != 0:
        raise RuntimeError("P14 training did not finish successfully")
    metrics = json.loads((BEST.parent / "metrics.json").read_text())
    if (sha(BEST) != EXPECTED_BEST_SHA or metrics.get("selected_checkpoint_sha256") != EXPECTED_BEST_SHA
            or metrics.get("selected_checkpoint") != "best.pt" or len(metrics.get("history", [])) != 8):
        raise RuntimeError("P14 selected checkpoint or full history drift")
    if read_training_lock(SHARED_LOCK) is not None or probe_running_trainers():
        raise RuntimeError("shared lock or other trainer is active")
    free_before = shutil.disk_usage(ROOT).free
    if free_before < 6 * 1024**3:
        raise RuntimeError("insufficient disk for bounded diagnostic")
    lease = TrainingLease.acquire(
        SHARED_LOCK, owner="20260930-p14-fold2-anchor01-diagnostic",
        config=str(ROOT / CONFIG), output_dir=str(OUTPUT), command=tuple(sys.argv),
    )
    started = time.time()
    code = 1
    try:
        command = [
            sys.executable, "-u", "scripts/diagnose_stage2_tta.py",
            "--experiment-config", CONFIG,
            "--checkpoint", str(BEST.relative_to(ROOT)),
            "--expected-checkpoint-sha256", EXPECTED_BEST_SHA,
            "--output-dir", str(OUTPUT.relative_to(ROOT)),
            "--batch-size", "64", "--workers", "4", "--four-view-scale384",
        ]
        with LOG.open("w", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                try:
                    code = child.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    free_now = shutil.disk_usage(ROOT).free
                    if (time.time() - started > LIMIT_SECONDS or free_before - free_now > MAX_ADDED_BYTES
                            or free_now < MIN_FREE_BYTES):
                        log.write("P14 diagnostic hard/resource cutoff; retain partial output.\n")
                        log.flush()
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait(timeout=30)
                        code = 124
                        break
        result = {
            "exit_code": code,
            "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "elapsed_seconds": time.time() - started,
            "disk_added_bytes": free_before - shutil.disk_usage(ROOT).free,
            "best_sha256": EXPECTED_BEST_SHA,
            "report_complete": (OUTPUT / "report.json").is_file(),
            "predictions_complete": (OUTPUT / "paired_predictions.csv").is_file(),
        }
        RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)
        return code
    finally:
        lease.release()


if __name__ == "__main__":
    raise SystemExit(main())
