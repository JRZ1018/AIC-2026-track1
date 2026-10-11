"""Bounded P18 crop-0.95 fold2 screen on the test-free RTX4090 server."""

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
from aictrack1.engine.training_queue import QueueItem, probe_running_trainers, read_training_lock, run_serial_queue  # noqa: E402

ORIGINAL = Path("/root/autodl-tmp/AICtrack1")
NAME = "20260930-p18-fold2-4090-crop095-screen"
QUEUE = ROOT / "artifacts/queues" / NAME
CONFIG = "configs/experiment/stage2_p18_fold2_sce_ac_crop095_8ep_4090.yaml"
CONTROL = "configs/experiment/stage2_p13_fold2_sce_ac_8ep_4090_control.yaml"
DATA = "configs/data/stage2_fold2_336_crop095.yaml"
CONTROL_DATA = "configs/data/stage2_fold2_336.yaml"
OUTPUT = "artifacts/experiments/20260930-stage2-p18-fold2-sce-ac-crop095-8ep-4090"
LOCK = ORIGINAL / "artifacts/queues/training.lock.json"
HARD_SECONDS = 3 * 3600
MAX_ADDED = 2 * 1024**3
MIN_FREE = 5 * 1024**3
MIN_START_FREE = 7 * 1024**3
EXPECTED = {
    "scripts/train_peft.py": "801a12cef8d402ab4db7dc39c28a9f90d3fd9f29a34bbc9140eaf21f1de79573",
    "src/aictrack1/methods/confidence.py": "ea76a610844e460373e20aced0ccbd236247a5cff95fa159288eb6183a561bf8",
    "src/aictrack1/methods/visual_ot.py": "4258e6f75e5c33ca4ddc892c92c7afe0c7860555400d4fb2c3bfa8743cc3f327",
    "artifacts/audits/stage2/train/samples.csv": "1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d",
    "artifacts/audits/stage2/train/split_fold2_seeded_random_20260924.csv": "1e5eb7e9adbc234ba933bf76c99896602b8bdce5557879f90224ae2bb4cf2a50",
    "artifacts/audits/stage2/train/label_conflicts.csv": "b1afe084e654aeac5d431501657734c168a3cfc1f928defacac219dcf84c430a",
    "artifacts/audits/stage2/metadata/classes.csv": "0f54c471945f356177c885ba2f9edc943d669a61f12c73503fd4632c91ef40be",
    "artifacts/experiments/20260925-stage2-fold2-c5-h1-336-seed20260921/best.pt": "83380620f06af442ddbe4e55f048d6519d03ce95e672ab3bc51469904b803500",
    "artifacts/weights/ViT-B-32.pt": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
}
INPUTS = (
    "scripts/run_stage2_p18_fold2_crop095_4090.py", "src/aictrack1/engine/training_queue.py",
    "src/aictrack1/data/transforms.py", "src/aictrack1/methods/sce.py",
    "configs/model/openai_clip_vit_b32.yaml", "requirements-lock.txt",
    CONFIG, CONTROL, DATA, CONTROL_DATA, *EXPECTED,
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def one_variable() -> None:
    control = yaml.safe_load((ROOT / CONTROL).read_text(encoding="utf-8"))["experiment"]
    candidate = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))["experiment"]
    for key in ("run_id", "output_dir", "dataset_config"):
        control.pop(key)
        candidate.pop(key)
    if candidate != control or candidate["training_scope"] != "fixed_split_validation" or candidate["training"]["epochs"] != 8:
        raise RuntimeError("P18 experiment changed beyond dataset config")
    old = yaml.safe_load((ROOT / CONTROL_DATA).read_text(encoding="utf-8"))["dataset"]
    new = yaml.safe_load((ROOT / DATA).read_text(encoding="utf-8"))["dataset"]
    if old["transforms"].pop("train_crop_scale_min") != 0.8 or new["transforms"].pop("train_crop_scale_min") != 0.95 or old != new:
        raise RuntimeError("P18 data config changed beyond train crop minimum")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if sys.platform != "linux" or ROOT != Path("/root/autodl-tmp/AICtrack1_p18_crop095"):
        raise RuntimeError("P18 requires exact test-free Linux workspace")
    if (ROOT / "DataSetStage2/test").exists() or QUEUE.exists() or (ROOT / OUTPUT).exists():
        raise RuntimeError("P18 test present or queue/output already exists")
    if read_training_lock(LOCK) is not None or probe_running_trainers():
        raise RuntimeError("shared lock or another trainer active")
    if not torch.cuda.is_available() or "RTX 4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("approved RTX4090 unavailable")
    gpu_free, _ = torch.cuda.mem_get_info(0)
    disk_free = shutil.disk_usage(ROOT).free
    if gpu_free < 20 * 1024**3 or disk_free < MIN_START_FREE:
        raise RuntimeError(f"P18 resource guard failed: gpu={gpu_free}, disk={disk_free}")
    missing = [name for name in INPUTS if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"P18 missing inputs: {missing}")
    frozen = {name: sha(ROOT / name) for name in INPUTS}
    drift = {name: digest for name, digest in EXPECTED.items() if frozen[name] != digest}
    if drift:
        raise RuntimeError(f"P18 source SHA mismatch: {drift}")
    one_variable()
    if sum(1 for _ in (ROOT / "DataSetStage2/train").rglob("*.jpg")) != 148695:
        raise RuntimeError("P18 official train count mismatch")
    state = {"scope": "Stage2 fold2 train-only crop-0.95 screen", "gpu": torch.cuda.get_device_name(0), "gpu_free": gpu_free, "disk_free": disk_free, "hard_seconds": HARD_SECONDS, "max_added_bytes": MAX_ADDED, "input_sha256": frozen}
    if not args.run:
        print(json.dumps(state, indent=2))
        return 0
    started = datetime.now(ZoneInfo("Asia/Shanghai"))
    deadline = started + timedelta(seconds=HARD_SECONDS)
    QUEUE.mkdir(parents=True, exist_ok=False)
    (QUEUE / "source_receipt.json").write_text(json.dumps({**state, "started_at": started.isoformat(), "deadline": deadline.isoformat(), "queue_pid": os.getpid()}, indent=2) + "\n", encoding="utf-8")

    def launch(command, cwd, output, on_start) -> int:
        if any(sha(ROOT / name) != digest for name, digest in frozen.items()):
            raise RuntimeError("P18 frozen input changed before launch")
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
                    output.write(f"P18 hard/resource cutoff: free={current_free}, added={disk_free-current_free}\n")
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
        items=[QueueItem("candidate_8ep", CONFIG, OUTPUT, "P18 crop minimum 0.95 SCE+AC fold2 screen")],
        launcher=launch, lock_path=LOCK,
    )
    (QUEUE / "queue_result.json").write_text(json.dumps({"exit_code": code, "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}, indent=2) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
