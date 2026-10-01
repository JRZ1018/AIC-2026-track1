"""Bounded Stage2 train-only P21 fold2 20-epoch schedule screen on RTX4090.

No-argument invocation is read-only preflight. --run consumes the approved batch.
"""

from __future__ import annotations

import argparse
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aictrack1.engine.training_queue import (  # noqa: E402
    QueueItem, probe_running_trainers, read_training_lock, run_serial_queue,
)

NAME = "20261001-p21-fold2-4090-20ep-screen"
EXPECTED_ROOT = Path("/root/autodl-tmp/AICtrack1_p21_20ep_20261001")
ORIGINAL = Path("/root/autodl-tmp/AICtrack1")
QUEUE = ROOT / "artifacts/queues" / NAME
LOCK = ORIGINAL / "artifacts/queues/training.lock.json"
CONFIG = "configs/experiment/stage2_p21_fold2_sce_ac_20ep_4090_screen.yaml"
OUTPUT = "artifacts/experiments/20261001-stage2-p21-fold2-sce-ac-20ep-4090-screen"
HARD_SECONDS = 3 * 3600 + 30 * 60
MAX_ADDED = 3 * 1024**3
MIN_START_FREE = 9 * 1024**3
MIN_FREE = 5 * 1024**3
EXPECTED_ASSETS = {
    "artifacts/weights/ViT-B-32.pt": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
    "artifacts/experiments/20260925-stage2-fold2-c5-h1-336-seed20260921/best.pt": "83380620f06af442ddbe4e55f048d6519d03ce95e672ab3bc51469904b803500",
    "artifacts/audits/stage2/train/samples.csv": "1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d",
    "artifacts/audits/stage2/train/split_fold2_seeded_random_20260924.csv": "1e5eb7e9adbc234ba933bf76c99896602b8bdce5557879f90224ae2bb4cf2a50",
    "artifacts/audits/stage2/train/label_conflicts.csv": "b1afe084e654aeac5d431501657734c168a3cfc1f928defacac219dcf84c430a",
    "artifacts/audits/stage2/metadata/classes.csv": "0f54c471945f356177c885ba2f9edc943d669a61f12c73503fd4632c91ef40be",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight() -> dict:
    if sys.platform != "linux" or ROOT != EXPECTED_ROOT:
        raise RuntimeError("P21 requires its exact isolated Linux workspace")
    if (ROOT / "DataSetStage2/test").exists() or QUEUE.exists() or (ROOT / OUTPUT).exists():
        raise RuntimeError("test, queue or output already exists")
    if read_training_lock(LOCK) is not None or probe_running_trainers():
        raise RuntimeError("shared lock or another trainer active")
    if not torch.cuda.is_available() or "RTX 4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("approved RTX4090 unavailable")
    disk_free = shutil.disk_usage(ROOT).free
    gpu_free, _ = torch.cuda.mem_get_info(0)
    if disk_free < MIN_START_FREE or gpu_free < 20 * 1024**3:
        raise RuntimeError(f"resource guard failed: gpu={gpu_free}, disk={disk_free}")
    config = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))["experiment"]
    train = config["training"]
    if (config["competition_stage"] != "stage2" or config["training_scope"] != "fixed_split_validation"
            or config["execution_enabled"] is not True or config["device"] != "cuda"
            or config["run_id"] != Path(OUTPUT).name or config["output_dir"] != "../../" + OUTPUT
            or config["base_loss"] != {"kind": "sce", "alpha": 0.1, "beta": 1.0, "log_zero": -4.0}
            or config["peft"] != {"mode": "last_n_blocks", "lora_blocks": 8}
            or train["epochs"] != 20 or train["batch_size"] != 64 or train["workers"] != 16
            or train["prefetch_factor"] != 4 or train["persistent_workers"] is not True
            or train["seed"] != 20260921):
        raise RuntimeError("P21 frozen recipe mismatch")
    expected = json.loads((ROOT / "p21_input_sha256.json").read_text(encoding="utf-8"))
    for name, digest in EXPECTED_ASSETS.items():
        if expected.get(name) != digest:
            raise RuntimeError(f"P21 provenance manifest mismatch: {name}")
    drift = [name for name, digest in expected.items()
             if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
    if drift:
        raise RuntimeError(f"P21 frozen source/input SHA mismatch: {drift}")
    train_root = ROOT / "DataSetStage2/train"
    if not train_root.is_dir() or sum(1 for _ in train_root.rglob("*.jpg")) != 148695:
        raise RuntimeError("official Stage2 train root/count mismatch")
    return {"scope": "Stage2 train-derived fold2 only; no test", "gpu": torch.cuda.get_device_name(0),
            "gpu_free": gpu_free, "disk_free": disk_free, "hard_seconds": HARD_SECONDS,
            "max_added_bytes": MAX_ADDED, "input_sha256": expected}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    receipt = preflight()
    if not args.run:
        print(json.dumps({k: v for k, v in receipt.items() if k != "input_sha256"}, indent=2))
        return 0
    started = datetime.now(ZoneInfo("Asia/Shanghai"))
    deadline = started + timedelta(seconds=HARD_SECONDS)
    QUEUE.mkdir(parents=True, exist_ok=False)
    receipt.update({"started_at": started.isoformat(), "deadline": deadline.isoformat(),
                    "queue_pid": os.getpid(), "authorization": "2026-10-01 user approved server if free"})
    (QUEUE / "source_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    def launch(command, cwd, output, on_start) -> int:
        if any(sha(ROOT / name) != digest for name, digest in receipt["input_sha256"].items()):
            raise RuntimeError("P21 frozen input changed")
        child = subprocess.Popen(list(command), cwd=cwd, stdout=output, stderr=subprocess.STDOUT,
                                 start_new_session=True)
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
                if time.time() >= deadline.timestamp() or free < MIN_FREE or receipt["disk_free"] - free > MAX_ADDED:
                    output.write(f"P21 hard/resource cutoff: free={free}, added={receipt['disk_free']-free}\n")
                    output.flush()
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=30)
                    return 124

    code = run_serial_queue(root=ROOT, python=Path(sys.executable), logs=QUEUE,
                            queue_name=NAME,
                            items=[QueueItem("candidate_20ep", CONFIG, OUTPUT, "P21 SCE+AC fold2 20-epoch schedule screen")],
                            launcher=launch, lock_path=LOCK)
    (QUEUE / "queue_result.json").write_text(json.dumps({"exit_code": code,
        "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}, indent=2) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
