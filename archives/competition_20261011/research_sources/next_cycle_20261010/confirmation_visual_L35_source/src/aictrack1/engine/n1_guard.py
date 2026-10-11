"""N1 source/input and resource checks; atomic leases remain in training_queue."""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from aictrack1.engine.training_queue import RunningTrainer, probe_running_trainers

SERVER_ROOT = Path("/root/autodl-tmp/AICtrack1_n1_weakac_20261002")
INPUT_ROOT = Path("/root/autodl-tmp/AICtrack1_q0_fold2_inputs_20261002")
INPUT_RECEIPT_SHA = "db0c45e3a361ac1fe9924083df5b403d6ed09ed3ee5ea316212000c5f0cd3137"
LOCK = Path("/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json")
NAME = "20261002-n1-fold2-4090-control20-vs-weak20"
GIB = 1024**3
INPUT_FILES = (
    "artifacts/audits/stage2/train/samples.csv",
    "artifacts/audits/stage2/train/split_fold2_seeded_random_20260924.csv",
    "artifacts/audits/stage2/train/label_conflicts.csv",
    "artifacts/audits/stage2/metadata/classes.csv",
    "artifacts/experiments/20260925-stage2-fold2-c5-h1-336-seed20260921/best.pt",
    "artifacts/weights/ViT-B-32.pt",
)


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def exact_server(root: Path) -> None:
    if sys.platform != "linux" or root.resolve() != SERVER_ROOT:
        raise RuntimeError("N1 runs only on the exact isolated teammate server root")
    for parent in (root, INPUT_ROOT):
        if os.path.lexists(parent / "DataSetStage2/test"):
            raise RuntimeError("test entry exists in N1/input copy")


def verify_bundle(root: Path) -> dict:
    manifest = json.loads((root / "n1_frozen_sha256.json").read_text())
    if manifest.get("schema") != "aictrack1.n1.frozen_bundle.v1":
        raise RuntimeError("invalid N1 source bundle")
    for relative, expected in manifest["files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not (root / path).resolve().is_relative_to(root.resolve()):
            raise RuntimeError("N1 source manifest escape")
        if sha(root / path) != expected:
            raise RuntimeError("N1 source drift: " + relative)
    return manifest


def verify_inputs(root: Path, *, staged: bool) -> dict:
    receipt_path = INPUT_ROOT / "source_receipt.json"
    if sha(receipt_path) != INPUT_RECEIPT_SHA:
        raise RuntimeError("isolated input receipt drift")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("status") != "train_only_inputs_complete" or receipt.get("train_rows") != 148695
            or receipt.get("validation_rows") != 29737 or not receipt.get("hardlink_inode_verified")
            or receipt.get("test_files_selected") != 0):
        raise RuntimeError("incomplete N1 train-only source")
    for relative in INPUT_FILES:
        expected = receipt["source_files"][relative]["sha256"]
        if sha(INPUT_ROOT / relative) != expected or (staged and sha(root / relative) != expected):
            raise RuntimeError("N1 train input drift: " + relative)
    return receipt


def exclusive_idle() -> list[RunningTrainer]:
    running = probe_running_trainers()
    query = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, timeout=20, check=True)
    for line in query.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2:
            continue
        pid, command = int(fields[0]), fields[1]
        executable = command.split(maxsplit=1)[0].rsplit("/", 1)[-1]
        if pid != os.getpid() and executable.startswith("python") and any(
                marker in command for marker in ("diagnose_", "_export", "_inference", "audit_q0_", "epoch_benchmark")):
            running.append(RunningTrainer(pid, command))
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=20, check=True)
    for line in apps.stdout.splitlines():
        if line.strip() and int(line.strip()) != os.getpid():
            running.append(RunningTrainer(int(line.strip()), "external GPU compute application"))
    return running


def resource_reason(initial_free: int, current_free: int, deadline: float, now: float) -> str | None:
    if now >= deadline:
        return "six-hour hard training deadline"
    if current_free < 7 * GIB:
        return "disk remainder below seven GiB"
    if initial_free - current_free > 6 * GIB:
        return "cumulative two-arm training disk increase above six GiB"
    return None


def config_path(arm: str) -> str:
    if arm not in ("control", "weak"):
        raise ValueError("unknown N1 arm")
    return f"configs/experiment/stage2_n1_fold2_{arm}20.yaml"


def output_path(arm: str) -> str:
    config_path(arm)
    return f"artifacts/experiments/20261002-stage2-n1-fold2-{arm}-20ep-seed20260921"
