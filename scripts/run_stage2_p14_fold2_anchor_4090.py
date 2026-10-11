"""Bounded P14 fold2 weight-anchor screen in an isolated server workspace.

Default is read-only preflight. --run uses the ORIGINAL project's atomic lock,
so this isolated checkout cannot overlap another training task on the instance.
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
    DEFAULT_LOCK_RELATIVE_PATH,
    QueueItem,
    probe_running_trainers,
    read_training_lock,
    run_serial_queue,
)

ORIGINAL = Path("/root/autodl-tmp/AICtrack1")
NAME = "20260930-p14-fold2-4090-anchor01-screen"
CONFIG = "configs/experiment/stage2_p14_fold2_sce_ac_anchor01_8ep_4090.yaml"
CONTROL_CONFIG = "configs/experiment/stage2_p13_fold2_sce_ac_8ep_4090_control.yaml"
OUTPUT = "artifacts/experiments/20260930-stage2-p14-fold2-sce-ac-anchor01-8ep-4090"
LOGS = ROOT / "artifacts/queues" / NAME
HARD_SECONDS = 3 * 3600
MAX_ADDED_BYTES = 2 * 1024**3
MIN_FREE_BYTES = 5 * 1024**3
EXPECTED_SHA = {
    "artifacts/audits/stage2/train/samples.csv": "1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d",
    "artifacts/audits/stage2/train/split_fold2_seeded_random_20260924.csv": "1e5eb7e9adbc234ba933bf76c99896602b8bdce5557879f90224ae2bb4cf2a50",
    "artifacts/audits/stage2/train/label_conflicts.csv": "b1afe084e654aeac5d431501657734c168a3cfc1f928defacac219dcf84c430a",
    "artifacts/audits/stage2/metadata/classes.csv": "0f54c471945f356177c885ba2f9edc943d669a61f12c73503fd4632c91ef40be",
    "artifacts/experiments/20260925-stage2-fold2-c5-h1-336-seed20260921/best.pt": "83380620f06af442ddbe4e55f048d6519d03ce95e672ab3bc51469904b803500",
    "artifacts/weights/ViT-B-32.pt": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
}
INPUTS = (
    "scripts/train_peft.py",
    "scripts/run_stage2_p14_fold2_anchor_4090.py",
    "src/aictrack1/engine/training_queue.py",
    "src/aictrack1/methods/weight_anchor.py",
    "src/aictrack1/methods/sce.py",
    "configs/data/stage2_fold2_336.yaml",
    "configs/model/openai_clip_vit_b32.yaml",
    CONFIG,
    CONTROL_CONFIG,
    "requirements-lock.txt",
    *EXPECTED_SHA,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def free_ram() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable unavailable")


def validate_one_variable() -> None:
    control = yaml.safe_load((ROOT / CONTROL_CONFIG).read_text(encoding="utf-8"))["experiment"]
    candidate = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))["experiment"]
    for key in ("run_id", "output_dir"):
        control.pop(key)
        candidate.pop(key)
    if candidate.pop("weight_anchor", None) != {"enabled": True, "strength": 0.1} or candidate != control:
        raise RuntimeError("P14 candidate differs from P13 fold2 control beyond weight anchor")
    if control["training_scope"] != "fixed_split_validation" or control["device"] != "cuda":
        raise RuntimeError("P14 must be Stage2 train-derived fold validation on CUDA")
    if control["training"]["epochs"] != 8:
        raise RuntimeError("P14 short screen requires exactly 8 epochs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if sys.platform != "linux" or ROOT != Path("/root/autodl-tmp/AICtrack1_p14_anchor01"):
        raise RuntimeError("P14 requires the exact isolated Linux workspace")
    if (ROOT / "DataSetStage2/test").exists():
        raise RuntimeError("isolated P14 workspace must not contain test")
    if LOGS.exists() or (ROOT / OUTPUT).exists():
        raise FileExistsError("P14 queue/output already exists")
    shared_lock = ORIGINAL / DEFAULT_LOCK_RELATIVE_PATH
    if read_training_lock(shared_lock) is not None or probe_running_trainers():
        raise RuntimeError("shared training lock or another trainer is active")
    if not torch.cuda.is_available() or "RTX 4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("approved RTX4090 unavailable")
    gpu_free, gpu_total = torch.cuda.mem_get_info(0)
    disk_free, ram_free = shutil.disk_usage(ROOT).free, free_ram()
    if gpu_free < 20 * 1024**3 or disk_free < 8 * 1024**3 or ram_free < 10 * 1024**3:
        raise RuntimeError(f"resource guard failed: gpu={gpu_free}, disk={disk_free}, ram={ram_free}")
    missing = [path for path in INPUTS if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"P14 inputs missing: {missing}")
    frozen = {path: sha256(ROOT / path) for path in INPUTS}
    drift = {path: expected for path, expected in EXPECTED_SHA.items() if frozen[path] != expected}
    if drift:
        raise RuntimeError(f"P14 source SHA mismatch: {drift}")
    validate_one_variable()
    train_root = ROOT / "DataSetStage2/train"
    if not train_root.is_dir() or sum(1 for _ in train_root.rglob("*.jpg")) != 148695:
        raise RuntimeError("P14 official train root/count mismatch")
    state = {
        "scope": "Stage2 fold2 train-derived one-variable anchor short screen; no test",
        "run": args.run,
        "shared_lock": str(shared_lock),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_free": gpu_free,
        "disk_free": disk_free,
        "ram_free": ram_free,
        "budget_seconds": HARD_SECONDS,
        "budget_added_bytes": MAX_ADDED_BYTES,
        "input_sha256": frozen,
    }
    if not args.run:
        print(json.dumps({key: value for key, value in state.items() if key != "input_sha256"}, indent=2))
        return 0
    started = datetime.now(ZoneInfo("Asia/Shanghai"))
    deadline = started + timedelta(seconds=HARD_SECONDS)
    LOGS.mkdir(parents=True, exist_ok=False)
    state.update({"started_at": started.isoformat(), "hard_deadline": deadline.isoformat(), "queue_pid": os.getpid()})
    (LOGS / "source_receipt.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    def launch(command, cwd, output, on_start) -> int:
        if any(sha256(ROOT / path) != digest for path, digest in frozen.items()):
            raise RuntimeError("P14 frozen inputs changed before launch")
        if time.time() >= deadline.timestamp():
            return 124
        child = subprocess.Popen(list(command), cwd=cwd, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            on_start(child.pid)
        except Exception:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=30)
            raise
        while True:
            try:
                return child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                current_free = shutil.disk_usage(ROOT).free
                added = disk_free - current_free
                if time.time() >= deadline.timestamp() or current_free < MIN_FREE_BYTES or added > MAX_ADDED_BYTES:
                    output.write(f"P14 hard/resource cutoff: free={current_free}, added={added}; keep partial output.\n")
                    output.flush()
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=30)
                    return 124

    code = run_serial_queue(
        root=ROOT,
        python=Path(sys.executable),
        logs=LOGS,
        queue_name=NAME,
        items=[QueueItem("candidate_8ep", CONFIG, OUTPUT, "P14 anchor strength0.1 single-variable fold2 screen")],
        launcher=launch,
        lock_path=shared_lock,
    )
    (LOGS / "queue_result.json").write_text(json.dumps({"exit_code": code, "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
