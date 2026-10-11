"""Resume the in-flight P18 fold2 crop-0.95 screen with a faster dataloader.

Why this exists
---------------
P18 ran with ``training.workers: 4`` and left the RTX 4090 idle most of the
time: measured GPU utilisation was 0 % in 17 of 20 samples over ten minutes,
because the loader could only feed ~206 samples/s while the train split needs
118,958 samples per epoch. The project's own benchmark shows the same pipeline
reaches ~347 samples/s at 16 workers (1.68x) -- a throughput-only change.

Why the worker count is not an algorithm change
-----------------------------------------------
``IndexedImageDataset.__getitem__`` builds the augmentation RNG from
``np.random.SeedSequence([stochastic_seed, epoch, record.sample_index])``, so a
sample receives the same augmentation no matter which worker loads it. Sample
order comes from the main-process sampler/generator. Worker count therefore only
affects throughput.

Why the override is an environment variable
-------------------------------------------
Checkpoints record ``config_sha256`` and ``train_peft.py --resume`` re-checks it.
Editing ``workers`` in the YAML would invalidate that hash and make the five
completed epochs unusable, so ``train_peft.py`` was taught to read
``AICTRACK1_DATALOADER_WORKERS`` / ``AICTRACK1_DATALOADER_PREFETCH`` from the
environment instead, leaving the experiment file byte-identical.

Default is a read-only preflight; ``--run`` performs the resume.
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

ROOT = Path("/root/autodl-tmp/AICtrack1_p18_crop095")
ORIGINAL = Path("/root/autodl-tmp/AICtrack1")
LOCK = ORIGINAL / "artifacts/queues/training.lock.json"

NAME = "20261001-p18-fold2-4090-crop095-resume-optloader"
LOGS = ROOT / "artifacts/queues" / NAME
CONFIG = "configs/experiment/stage2_p18_fold2_sce_ac_crop095_8ep_4090.yaml"
OUTPUT = "artifacts/experiments/20260930-stage2-p18-fold2-sce-ac-crop095-8ep-4090"
RESUME = OUTPUT + "/last.pt"

WORKERS = 16
PREFETCH = 6
HARD_SECONDS = 2 * 3600
MIN_FREE = 5 * 1024**3
MAX_ADDED = 2 * 1024**3

sys.path.insert(0, str(ROOT / "src"))
from aictrack1.engine.training_queue import (  # noqa: E402
    QueueItem,
    probe_running_trainers,
    read_training_lock,
    run_serial_queue,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stop_owned_process(process: subprocess.Popen, output) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    output.write("resume cutoff: SIGTERM to trainer process group; retaining atomic last.pt\n")
    output.flush()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def preflight() -> dict:
    if os.name != "posix" or sys.platform != "linux":
        raise RuntimeError("this resume requires Linux")
    if LOGS.exists():
        raise FileExistsError(f"resume queue output already exists: {LOGS}")
    if read_training_lock(LOCK) is not None:
        raise RuntimeError(f"training lock still held: {LOCK}")
    running = probe_running_trainers()
    if running:
        raise RuntimeError(f"another trainer is active: {running}")
    if not torch.cuda.is_available() or "RTX 4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("approved RTX 4090 unavailable")

    gpu_free, gpu_total = torch.cuda.mem_get_info(0)
    disk_free = shutil.disk_usage(ROOT).free
    if gpu_free < 20 * 1024**3:
        raise RuntimeError(f"resume GPU guard failed: {gpu_free}")
    if disk_free < max(MIN_FREE, 5 * 1024**3):
        raise RuntimeError(f"resume disk guard failed: {disk_free}")

    config_path = ROOT / CONFIG
    resume_path = ROOT / RESUME
    output_dir = ROOT / OUTPUT
    if not config_path.is_file() or not resume_path.is_file():
        raise FileNotFoundError("config or resume checkpoint missing")
    if not output_dir.is_dir():
        raise FileNotFoundError(f"output dir missing: {output_dir}")
    if (output_dir / "metrics.json").exists() or (output_dir / "best.pt").exists():
        raise RuntimeError("output already looks complete; refusing to resume")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))["experiment"]
    if raw["training_scope"] != "fixed_split_validation" or raw["training"]["epochs"] != 8:
        raise RuntimeError("training scope or total schedule drift")
    if raw["output_dir"] != "../../" + OUTPUT or raw["execution_enabled"] is not True:
        raise RuntimeError("run identity or execution approval drift")

    config_sha = sha256(config_path)
    resume_sha = sha256(resume_path)

    state = torch.load(resume_path, map_location="cpu", weights_only=True)
    epoch = int(state.get("epoch", -1))
    history_len = len(state.get("history", ()))
    if epoch != history_len or epoch < 1:
        raise RuntimeError(f"resume checkpoint is not a clean epoch boundary: epoch={epoch}, history={history_len}")
    if epoch >= 8:
        raise RuntimeError("resume checkpoint already at the final epoch")
    if state.get("config_sha256") != config_sha:
        raise RuntimeError("resume checkpoint config_sha256 does not match the unchanged YAML")
    for key in ("trainable_state", "best_state", "optimizer", "scheduler", "scaler",
                "python_rng_state", "numpy_rng_state", "torch_rng_state", "confidence_weighting"):
        if key not in state:
            raise RuntimeError(f"resume checkpoint missing {key}")
    history_tail = state["history"][-1] if history_len else {}
    del state

    return {
        "scope": "P18 fold2 crop-0.95 screen, throughput-only dataloader resume",
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_free": gpu_free,
        "gpu_total": gpu_total,
        "disk_free": disk_free,
        "config": CONFIG,
        "config_sha256": config_sha,
        "resume_checkpoint": RESUME,
        "resume_sha256": resume_sha,
        "resume_epoch": epoch,
        "resume_history_len": history_len,
        "epochs_total": 8,
        "epochs_remaining": 8 - epoch,
        "last_epoch_top1": history_tail.get("top1"),
        "dataloader_override": {
            "AICTRACK1_DATALOADER_WORKERS": WORKERS,
            "AICTRACK1_DATALOADER_PREFETCH": PREFETCH,
            "baseline_workers_in_yaml": raw["training"]["workers"],
        },
        "hard_seconds": HARD_SECONDS,
        "max_added_bytes": MAX_ADDED,
        "training_lock": str(LOCK),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="launch the resume")
    args = parser.parse_args()

    receipt = preflight()
    if not args.run:
        print(json.dumps(receipt, indent=2))
        return 0

    started = datetime.now(ZoneInfo("Asia/Shanghai"))
    deadline = started + timedelta(seconds=HARD_SECONDS)
    start_free = receipt["disk_free"]
    LOGS.mkdir(parents=True, exist_ok=False)
    receipt.update({"started_at": started.isoformat(),
                    "hard_deadline": deadline.isoformat(),
                    "queue_pid": os.getpid()})
    (LOGS / "source_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    config_path = ROOT / CONFIG
    frozen_config_sha = receipt["config_sha256"]

    def launcher(command, cwd, output, on_start) -> int:
        if sha256(config_path) != frozen_config_sha:
            raise RuntimeError("experiment YAML changed after preflight")
        env = dict(os.environ)
        env["AICTRACK1_DATALOADER_WORKERS"] = str(WORKERS)
        env["AICTRACK1_DATALOADER_PREFETCH"] = str(PREFETCH)
        process = subprocess.Popen(list(command), cwd=cwd, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True, env=env)
        try:
            on_start(process.pid)
        except Exception:
            stop_owned_process(process, output)
            raise
        while True:
            try:
                return process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                free = shutil.disk_usage(ROOT).free
                if (time.time() >= deadline.timestamp() or free < MIN_FREE
                        or start_free - free > MAX_ADDED):
                    output.write(f"resume cutoff: free={free}\n")
                    output.flush()
                    stop_owned_process(process, output)
                    return 124

    item = QueueItem("candidate_8ep_resume", CONFIG, OUTPUT,
                     f"throughput-only resume from epoch {receipt['resume_epoch']} "
                     f"with {WORKERS} dataloader workers",
                     resume_checkpoint=RESUME)
    code = run_serial_queue(root=ROOT, python=Path(sys.executable), logs=LOGS,
                            queue_name=NAME, items=[item], launcher=launcher,
                            lock_path=LOCK)
    (LOGS / "queue_result.json").write_text(json.dumps({
        "exit_code": code,
        "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
