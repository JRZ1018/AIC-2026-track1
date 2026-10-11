"""Bounded Linux/4090 P13 fold2 8-vs-14 epoch confirmation queue.

Default mode is read-only preflight. --run consumes the explicitly approved
five-hour training window and never resumes or clears an existing lock.
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

NAME = "20260929-p13-fold2-4090-schedule-pair"
LOGS = ROOT / "artifacts/queues" / NAME
CONFIGS = (
    "configs/experiment/stage2_p13_fold2_sce_ac_8ep_4090_control.yaml",
    "configs/experiment/stage2_p13_fold2_sce_ac_14ep_4090_candidate.yaml",
)
OUTPUTS = (
    "artifacts/experiments/20260929-stage2-p13-fold2-sce-ac-8ep-4090-control",
    "artifacts/experiments/20260929-stage2-p13-fold2-sce-ac-14ep-4090-candidate",
)
INPUTS = (
    "scripts/train_peft.py",
    "scripts/run_stage2_p13_fold2_4090_schedule_pair.py",
    "src/aictrack1/engine/training_queue.py",
    "src/aictrack1/methods/sce.py",
    "configs/data/stage2_fold2_336.yaml",
    "configs/model/openai_clip_vit_b32.yaml",
    *CONFIGS,
    "artifacts/audits/stage2/train/samples.csv",
    "artifacts/audits/stage2/train/split_fold2_seeded_random_20260924.csv",
    "artifacts/audits/stage2/train/label_conflicts.csv",
    "artifacts/audits/stage2/metadata/classes.csv",
    "artifacts/experiments/20260925-stage2-fold2-c5-h1-336-seed20260921/best.pt",
    "artifacts/weights/ViT-B-32.pt",
)
EXPECTED_SHA = {
    "artifacts/audits/stage2/train/samples.csv": "1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d",
    "artifacts/audits/stage2/train/split_fold2_seeded_random_20260924.csv": "1e5eb7e9adbc234ba933bf76c99896602b8bdce5557879f90224ae2bb4cf2a50",
    "artifacts/audits/stage2/train/label_conflicts.csv": "b1afe084e654aeac5d431501657734c168a3cfc1f928defacac219dcf84c430a",
    "artifacts/audits/stage2/metadata/classes.csv": "0f54c471945f356177c885ba2f9edc943d669a61f12c73503fd4632c91ef40be",
    "artifacts/experiments/20260925-stage2-fold2-c5-h1-336-seed20260921/best.pt": "83380620f06af442ddbe4e55f048d6519d03ce95e672ab3bc51469904b803500",
    "artifacts/weights/ViT-B-32.pt": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
}
HARD_SECONDS = 5 * 3600
MAX_ADDED_BYTES = 4 * 1024**3
MIN_FREE_BYTES = 5 * 1024**3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def free_ram() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("cannot read Linux MemAvailable")


def validate_configs() -> None:
    first, second = [yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))["experiment"] for path in CONFIGS]
    if {key for key in first if first[key] != second[key]} != {"run_id", "output_dir", "training"}:
        raise RuntimeError("P13 fold2 config drift beyond run identity, output, and training")
    if {key for key in first["training"] if first["training"][key] != second["training"][key]} != {"epochs"}:
        raise RuntimeError("P13 fold2 training drift beyond epochs")
    if (first["training"]["epochs"], second["training"]["epochs"]) != (8, 14):
        raise RuntimeError("P13 fold2 must compare 8 versus 14 epochs")
    for config, output in zip((first, second), OUTPUTS, strict=True):
        if config["execution_enabled"] is not True or (ROOT / output).exists():
            raise RuntimeError("P13 fold2 output exists or approval state invalid")
        if config["output_dir"] != "../../" + output:
            raise RuntimeError("P13 fold2 output directory drift")
        if config["training_scope"] != "fixed_split_validation" or config["device"] != "cuda":
            raise RuntimeError("P13 fold2 scope/device drift")
        if config["split"] != "../../artifacts/audits/stage2/train/split_fold2_seeded_random_20260924.csv":
            raise RuntimeError("P13 fold2 split drift")
        if config["base_loss"] != {"kind": "sce", "alpha": 0.1, "beta": 1.0, "log_zero": -4.0}:
            raise RuntimeError("P13 fold2 SCE recipe drift")
        if config["training"]["gradient_clip_norm"] != 1.0:
            raise RuntimeError("P13 fold2 clip norm must match fold3 policy")


def stop_owned_process(process: subprocess.Popen, output) -> None:
    if process.poll() is not None:
        return
    output.write("Stopping only this queue-owned Linux process group.\n")
    output.flush()
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="launch approved GPU pair; default is read-only preflight")
    args = parser.parse_args()
    if os.name != "posix" or sys.platform != "linux":
        raise RuntimeError("this queue requires Linux")
    validate_configs()
    if LOGS.exists():
        raise FileExistsError(f"queue output already exists: {LOGS}")
    if read_training_lock(ROOT / DEFAULT_LOCK_RELATIVE_PATH) is not None or probe_running_trainers():
        raise RuntimeError("training lock or another supported trainer is active")
    if not torch.cuda.is_available() or "RTX 4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("P13 fold2 requires the approved RTX 4090")
    gpu_free, gpu_total = torch.cuda.mem_get_info(0)
    disk_free, ram_free = shutil.disk_usage(ROOT).free, free_ram()
    if gpu_free < 20 * 1024**3 or disk_free < 9 * 1024**3 or ram_free < 10 * 1024**3:
        raise RuntimeError(f"resource guard failed: gpu={gpu_free}, disk={disk_free}, ram={ram_free}")
    missing = [path for path in INPUTS if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"P13 fold2 inputs missing: {missing}")
    frozen = {path: sha256(ROOT / path) for path in INPUTS}
    drift = {path: digest for path, digest in EXPECTED_SHA.items() if frozen[path] != digest}
    if drift:
        raise RuntimeError(f"P13 fold2 frozen provenance mismatch: {drift}")
    train_root = ROOT / "DataSetStage2/train"
    if not train_root.is_dir() or sum(1 for _ in train_root.rglob("*.jpg")) != 148695:
        raise RuntimeError("Stage2 train image root/count mismatch")
    state = {"scope": "Stage2 fold2 train-only matched 4090 SCE schedule pair", "run_enabled": args.run,
             "gpu_name": torch.cuda.get_device_name(0), "gpu_free": gpu_free, "gpu_total": gpu_total,
             "disk_free": disk_free, "ram_free": ram_free, "budget_training_seconds": HARD_SECONDS,
             "budget_added_bytes": MAX_ADDED_BYTES, "input_sha256": frozen}
    if not args.run:
        print(json.dumps({key: value for key, value in state.items() if key != "input_sha256"}, indent=2))
        return 0

    def verify_frozen() -> None:
        changed = [path for path, digest in frozen.items() if sha256(ROOT / path) != digest]
        if changed:
            raise RuntimeError(f"P13 fold2 frozen inputs changed: {changed}")

    started = datetime.now(ZoneInfo("Asia/Shanghai"))
    deadline = started + timedelta(seconds=HARD_SECONDS)
    LOGS.mkdir(parents=True, exist_ok=False)
    state.update({"started_at": started.isoformat(), "hard_deadline": deadline.isoformat(), "queue_pid": os.getpid()})
    (LOGS / "source_receipt.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    def bounded_launcher(command, cwd, output, on_start) -> int:
        verify_frozen()
        if time.time() >= deadline.timestamp():
            return 124
        process = subprocess.Popen(list(command), cwd=cwd, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            on_start(process.pid)
        except Exception:
            stop_owned_process(process, output)
            raise
        while True:
            try:
                return process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                current_free = shutil.disk_usage(ROOT).free
                added = disk_free - current_free
                if time.time() >= deadline.timestamp() or current_free < MIN_FREE_BYTES or added > MAX_ADDED_BYTES:
                    output.write(f"P13 fold2 hard/resource cutoff: free={current_free}, added={added}; retaining atomic last.pt.\n")
                    output.flush()
                    stop_owned_process(process, output)
                    return 124

    items = [QueueItem("control_8ep", CONFIGS[0], OUTPUTS[0], "same-server 8-epoch control"),
             QueueItem("candidate_14ep", CONFIGS[1], OUTPUTS[1], "14-epoch schedule confirmation")]
    code = run_serial_queue(root=ROOT, python=Path(sys.executable), logs=LOGS, queue_name=NAME,
                            items=items, launcher=bounded_launcher)
    verify_frozen()
    (LOGS / "queue_result.json").write_text(json.dumps({"exit_code": code,
        "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}, indent=2) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
