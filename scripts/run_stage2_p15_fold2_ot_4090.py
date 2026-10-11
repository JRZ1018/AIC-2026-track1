"""Bounded P15 fold2 OT-weighted SCE+AC screen on the test-free RTX4090 copy."""

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
from aictrack1.engine.training_queue import QueueItem, read_training_lock, probe_running_trainers, run_serial_queue  # noqa: E402

ORIGINAL = Path("/root/autodl-tmp/AICtrack1")
NAME = "20260930-p15-fold2-4090-visual-ot-screen"
QUEUE = ROOT / "artifacts/queues" / NAME
CONTROL = "configs/experiment/stage2_p13_fold2_sce_ac_8ep_4090_control.yaml"
CONFIG = "configs/experiment/stage2_p15_fold2_sce_ac_visual_ot_8ep_4090.yaml"
OUTPUT = "artifacts/experiments/20260930-stage2-p15-fold2-sce-ac-visual-ot-8ep-4090"
LOCK = ORIGINAL / "artifacts/queues/training.lock.json"
HARD_SECONDS = 3 * 3600
MAX_ADDED = 2 * 1024**3
MIN_FREE = 5 * 1024**3
EXPECTED = {
    "artifacts/audits/stage2/train/samples.csv": "1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d",
    "artifacts/audits/stage2/train/split_fold2_seeded_random_20260924.csv": "1e5eb7e9adbc234ba933bf76c99896602b8bdce5557879f90224ae2bb4cf2a50",
    "artifacts/audits/stage2/train/label_conflicts.csv": "b1afe084e654aeac5d431501657734c168a3cfc1f928defacac219dcf84c430a",
    "artifacts/audits/stage2/metadata/classes.csv": "0f54c471945f356177c885ba2f9edc943d669a61f12c73503fd4632c91ef40be",
    "artifacts/experiments/20260925-stage2-fold2-c5-h1-336-seed20260921/best.pt": "83380620f06af442ddbe4e55f048d6519d03ce95e672ab3bc51469904b803500",
    "artifacts/weights/ViT-B-32.pt": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
    "artifacts/features/stage2/p15_train_official336_20260930/complete.json": "6f0c3d2811b29b045dad4c6a27c8b01364a0daf2f00c600773ce9ffae12e459a",
}
INPUTS = (
    "scripts/train_peft.py", "scripts/run_stage2_p15_fold2_ot_4090.py",
    "src/aictrack1/methods/visual_ot.py", "src/aictrack1/methods/confidence.py",
    "src/aictrack1/engine/training_queue.py", "src/aictrack1/methods/sce.py",
    "configs/data/stage2_fold2_336.yaml", "configs/model/openai_clip_vit_b32.yaml",
    CONFIG, CONTROL, "requirements-lock.txt", *EXPECTED,
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_one_variable() -> None:
    control = yaml.safe_load((ROOT / CONTROL).read_text(encoding="utf-8"))["experiment"]
    candidate = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))["experiment"]
    for key in ("run_id", "output_dir"):
        control.pop(key)
        candidate.pop(key)
    expected_ot = {
        "enabled": True,
        "feature_cache": "../../artifacts/features/stage2/p15_train_official336_20260930",
        "temperature": 0.05, "iterations": 30,
        "low_trust_fraction": 0.1, "min_weight": 0.5,
    }
    if candidate.pop("visual_ot_auxiliary_weighting", None) != expected_ot or candidate != control:
        raise RuntimeError("P15 config differs from matched P13 control beyond OT auxiliary weighting")
    if control["training_scope"] != "fixed_split_validation" or control["training"]["epochs"] != 8 or control["device"] != "cuda":
        raise RuntimeError("P15 screen scope/epochs/device mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if sys.platform != "linux" or ROOT != Path("/root/autodl-tmp/AICtrack1_p15_ot01"):
        raise RuntimeError("P15 screen requires exact isolated Linux project")
    if (ROOT / "DataSetStage2/test").exists():
        raise RuntimeError("P15 screen refuses a workspace containing test")
    if QUEUE.exists() or (ROOT / OUTPUT).exists():
        raise FileExistsError("P15 queue/output already exists; no automatic retry")
    if read_training_lock(LOCK) is not None or probe_running_trainers():
        raise RuntimeError("shared training lock or another trainer active")
    if not torch.cuda.is_available() or "RTX 4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("approved RTX4090 unavailable")
    gpu_free, _ = torch.cuda.mem_get_info(0)
    disk_free = shutil.disk_usage(ROOT).free
    if gpu_free < 20 * 1024**3 or disk_free < 7 * 1024**3:
        raise RuntimeError(f"P15 resource guard failed: gpu={gpu_free}, disk={disk_free}")
    missing = [name for name in INPUTS if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"P15 inputs missing: {missing}")
    frozen = {name: sha(ROOT / name) for name in INPUTS}
    drift = {name: digest for name, digest in EXPECTED.items() if frozen[name] != digest}
    if drift:
        raise RuntimeError(f"P15 source SHA mismatch: {drift}")
    validate_one_variable()
    if sum(1 for _ in (ROOT / "DataSetStage2/train").rglob("*.jpg")) != 148695:
        raise RuntimeError("P15 official Stage2 train count mismatch")
    if not args.run:
        print(json.dumps({"scope": "Stage2 fold2 train-derived OT screen; no test", "gpu": torch.cuda.get_device_name(0), "gpu_free": gpu_free, "disk_free": disk_free, "budget_seconds": HARD_SECONDS, "max_added_bytes": MAX_ADDED, "input_sha256": frozen}, indent=2))
        return 0
    started = datetime.now(ZoneInfo("Asia/Shanghai"))
    deadline = started + timedelta(seconds=HARD_SECONDS)
    QUEUE.mkdir(parents=True, exist_ok=False)
    (QUEUE / "source_receipt.json").write_text(json.dumps({"started_at": started.isoformat(), "deadline": deadline.isoformat(), "queue_pid": os.getpid(), "input_sha256": frozen, "disk_free_start": disk_free}, indent=2) + "\n", encoding="utf-8")

    def launch(command, cwd, output, on_start) -> int:
        if any(sha(ROOT / name) != digest for name, digest in frozen.items()):
            raise RuntimeError("P15 frozen input changed before launch")
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
                current_free = shutil.disk_usage(ROOT).free
                if time.time() >= deadline.timestamp() or current_free < MIN_FREE or disk_free - current_free > MAX_ADDED:
                    output.write(f"P15 hard/resource cutoff: free={current_free}, added={disk_free-current_free}\n")
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
        items=[QueueItem("candidate_8ep", CONFIG, OUTPUT, "P15 static OT auxiliary weighting on SCE+AC")],
        launcher=launch, lock_path=LOCK,
    )
    (QUEUE / "queue_result.json").write_text(json.dumps({"exit_code": code, "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}, indent=2) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
