"""Same-source P21 fold2 fourteen-epoch control; Stage2 train only.

No-argument invocation is read-only preflight. ``--run`` consumes this single
bounded batch under the original project-wide atomic training lock.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aictrack1.engine.training_queue import (  # noqa: E402
    QueueItem, probe_running_trainers, read_training_lock, run_serial_queue,
)
from run_stage2_p21_fold2_4090_20ep import EXPECTED_ASSETS, sha  # noqa: E402

NAME = "20261001-p21-fold2-4090-14ep-matched"
EXPECTED_ROOT = Path("/root/autodl-tmp/AICtrack1_p21_fold2_control_20261001")
LOCK = Path("/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json")
QUEUE = ROOT / "artifacts/queues" / NAME
CONFIG = "configs/experiment/stage2_p21_fold2_sce_ac_14ep_4090_matched.yaml"
OUTPUT = "artifacts/experiments/20261001-stage2-p21-fold2-sce-ac-14ep-4090-matched"
HARD_SECONDS = 3 * 3600
MAX_ADDED = 3 * 1024**3
MIN_FREE = 5 * 1024**3


def preflight() -> dict:
    if sys.platform != "linux" or ROOT != EXPECTED_ROOT or (ROOT / "DataSetStage2/test").exists():
        raise RuntimeError("P21 control requires the exact train-only isolated root")
    if QUEUE.exists() or (ROOT / OUTPUT).exists() or LOCK.exists() or probe_running_trainers():
        raise RuntimeError("control output, another trainer, or shared lock exists")
    if not torch.cuda.is_available() or "RTX 4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("approved RTX4090 unavailable")
    disk_free = shutil.disk_usage(ROOT).free
    gpu_free, _ = torch.cuda.mem_get_info(0)
    if disk_free < 9 * 1024**3 or gpu_free < 20 * 1024**3:
        raise RuntimeError("control resource guard failed")
    candidate = yaml.safe_load((ROOT / "configs/experiment/stage2_p21_fold2_sce_ac_20ep_4090_screen.yaml")
                               .read_text(encoding="utf-8"))["experiment"]
    control = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))["experiment"]
    expected = dict(candidate)
    expected.update({"run_id": Path(OUTPUT).name, "output_dir": "../../" + OUTPUT,
                     "training": {**candidate["training"], "epochs": 14}})
    if control != expected:
        raise RuntimeError("control must differ from P21 candidate only in run/output/epochs")
    frozen = json.loads((ROOT / "p21_input_sha256.json").read_text(encoding="utf-8"))
    if any(frozen.get(name) != digest for name, digest in EXPECTED_ASSETS.items()):
        raise RuntimeError("P21 official input manifest changed")
    drift = [name for name, digest in frozen.items() if not (ROOT / name).is_file()
             or sha(ROOT / name) != digest]
    if drift:
        raise RuntimeError(f"P21 core source/input SHA drift: {drift}")
    if sum(1 for _ in (ROOT / "DataSetStage2/train").rglob("*.jpg")) != 148695:
        raise RuntimeError("Stage2 official train count changed")
    return {"scope": "Stage2 train-derived fold2 matched control; no test",
            "disk_free": disk_free, "gpu_free": gpu_free, "hard_seconds": HARD_SECONDS,
            "max_added_bytes": MAX_ADDED, "core_source_sha256": frozen,
            "candidate_config_sha256": sha(ROOT / "configs/experiment/stage2_p21_fold2_sce_ac_20ep_4090_screen.yaml"),
            "control_config_sha256": sha(ROOT / CONFIG)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    receipt = preflight()
    if not args.run:
        print(json.dumps({k: v for k, v in receipt.items() if k != "core_source_sha256"}, indent=2))
        return 0
    started = datetime.now(ZoneInfo("Asia/Shanghai"))
    deadline = started + timedelta(seconds=HARD_SECONDS)
    QUEUE.mkdir(parents=True, exist_ok=False)
    receipt.update({"started_at": started.isoformat(), "deadline": deadline.isoformat(),
                    "queue_pid": os.getpid()})
    (QUEUE / "source_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    def launch(command, cwd, output, on_start) -> int:
        if any(sha(ROOT / name) != digest for name, digest in receipt["core_source_sha256"].items()):
            raise RuntimeError("P21 frozen input changed before matched control")
        child = subprocess.Popen(list(command), cwd=cwd, stdout=output,
                                 stderr=subprocess.STDOUT, start_new_session=True)
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
                if (time.time() >= deadline.timestamp() or free < MIN_FREE
                        or receipt["disk_free"] - free > MAX_ADDED):
                    output.write(f"P21 control hard/resource cutoff: free={free}\n")
                    output.flush()
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=30)
                    return 124

    code = run_serial_queue(root=ROOT, python=Path(sys.executable), logs=QUEUE,
                            queue_name=NAME, lock_path=LOCK,
                            items=[QueueItem("control_14ep", CONFIG, OUTPUT,
                                             "P21 matched fold2 SCE+AC fourteen epochs")],
                            launcher=launch)
    (QUEUE / "queue_result.json").write_text(json.dumps({"exit_code": code,
        "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
