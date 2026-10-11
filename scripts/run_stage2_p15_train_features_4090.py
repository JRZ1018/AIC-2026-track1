"""Bounded server-only Stage2 train feature extraction for P15 feasibility audit."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aictrack1.engine.training_queue import TrainingLease, probe_running_trainers, read_training_lock  # noqa: E402

NAME = "20260930-p15-4090-train-features"
QUEUE = ROOT / "artifacts/queues" / NAME
OUTPUT = ROOT / "artifacts/features/stage2/p15_train_official336_20260930"
LOCK = Path("/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json")
HARD_SECONDS = 90 * 60
MAX_ADDED = 1024**3
MIN_FREE = 6 * 1024**3
EXPECTED = {
    "artifacts/audits/stage2/train/samples.csv": "1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d",
    "artifacts/audits/stage2/train/split_fold2_seeded_random_20260924.csv": "1e5eb7e9adbc234ba933bf76c99896602b8bdce5557879f90224ae2bb4cf2a50",
    "artifacts/weights/ViT-B-32.pt": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
}
INPUTS = (
    "scripts/extract_clip_features.py",
    "scripts/run_stage2_p15_train_features_4090.py",
    "src/aictrack1/engine/training_queue.py",
    "configs/data/stage2_fold2_336.yaml",
    "configs/model/openai_clip_vit_b32.yaml",
    *EXPECTED,
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if sys.platform != "linux" or ROOT != Path("/root/autodl-tmp/AICtrack1_p14_anchor01"):
        raise RuntimeError("P15 feature extraction requires the exact verified Linux project")
    if (ROOT / "DataSetStage2/test").exists():
        raise RuntimeError("P15 train-only server must not contain test")
    if QUEUE.exists() or OUTPUT.exists():
        raise FileExistsError("P15 queue/output already exists; no automatic retry")
    if read_training_lock(LOCK) is not None or probe_running_trainers():
        raise RuntimeError("shared lock or trainer already active")
    if not torch.cuda.is_available() or "RTX 4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("verified RTX4090 unavailable")
    gpu_free, _ = torch.cuda.mem_get_info(0)
    disk_free = shutil.disk_usage(ROOT).free
    if gpu_free < 20 * 1024**3 or disk_free < 7 * 1024**3:
        raise RuntimeError(f"resource guard failed: gpu_free={gpu_free}, disk_free={disk_free}")
    missing = [name for name in INPUTS if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing P15 input: {missing}")
    frozen = {name: sha(ROOT / name) for name in INPUTS}
    drift = {name: expected for name, expected in EXPECTED.items() if frozen[name] != expected}
    if drift:
        raise RuntimeError(f"source SHA mismatch: {drift}")
    if sum(1 for _ in (ROOT / "DataSetStage2/train").rglob("*.jpg")) != 148695:
        raise RuntimeError("Stage2 train count mismatch")
    if not args.run:
        print(json.dumps({"scope": "Stage2 train only, no test", "gpu": torch.cuda.get_device_name(0), "gpu_free": gpu_free, "disk_free": disk_free, "input_sha256": frozen}, indent=2))
        return 0
    lease = TrainingLease.acquire(LOCK, owner=NAME, config="configs/data/stage2_fold2_336.yaml", output_dir=str(OUTPUT), command=sys.argv)
    started = datetime.now(ZoneInfo("Asia/Shanghai"))
    deadline = started + timedelta(seconds=HARD_SECONDS)
    try:
        QUEUE.mkdir(parents=True, exist_ok=False)
        receipt = {"scope": "Stage2 train-only P15 features", "started_at": started.isoformat(), "deadline": deadline.isoformat(), "queue_pid": os.getpid(), "lock_token": lease.record.token, "input_sha256": frozen, "disk_free_start": disk_free, "max_added_bytes": MAX_ADDED, "min_free_bytes": MIN_FREE}
        (QUEUE / "source_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        command = [sys.executable, "scripts/extract_clip_features.py", "--config", "configs/data/stage2_fold2_336.yaml", "--source", "train", "--manifest", "artifacts/audits/stage2/train/samples.csv", "--split", "artifacts/audits/stage2/train/split_fold2_seeded_random_20260924.csv", "--model-config", "configs/model/openai_clip_vit_b32.yaml", "--visual-resolution", "336", "--device", "cuda", "--feature-dtype", "float16", "--output", str(OUTPUT), "--batch-size", "64", "--workers", "4", "--shard-size", "4096"]
        with (QUEUE / "extract.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            lease.attach_trainer(child.pid)
            while True:
                try:
                    code = child.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    current_free = shutil.disk_usage(ROOT).free
                    if time.time() >= deadline.timestamp() or current_free < MIN_FREE or disk_free - current_free > MAX_ADDED:
                        log.write(f"P15 hard/resource cutoff: free={current_free}, added={disk_free-current_free}\n")
                        log.flush()
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait(timeout=30)
                        code = 124
                        break
        if code == 0:
            stats = json.loads((OUTPUT / "extraction_stats.json").read_text(encoding="utf-8"))
            if stats.get("record_count") != 148695 or not (OUTPUT / "complete.json").is_file():
                raise RuntimeError("P15 feature cache incomplete despite exit zero")
        result = {"exit_code": code, "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "disk_free_end": shutil.disk_usage(ROOT).free}
        (QUEUE / "queue_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return code
    finally:
        lease.release()


if __name__ == "__main__":
    raise SystemExit(main())
