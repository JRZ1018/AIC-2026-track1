"""Matched local fold2 SCE vs NCE+RCE screen under the shared serial lease."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import (  # noqa: E402
    QueueItem,
    TrainingLease,
    probe_running_trainers,
    read_training_lock,
    run_serial_queue,
)


QUEUE_NAME = "20260929-p11-local-fold2-nce-rce-pair"
QUEUE_DIR = ROOT / "artifacts/queues" / QUEUE_NAME
CONTROL = "configs/experiment/stage2_p11_fold2_sce_control.yaml"
CANDIDATE = "configs/experiment/stage2_p11_fold2_nce_rce_candidate.yaml"
SOURCE_INPUTS = (
    "scripts/train_peft.py",
    "src/aictrack1/engine/training_queue.py",
    "src/aictrack1/methods/nce_rce.py",
    "src/aictrack1/methods/sce.py",
    "artifacts/audits/stage2/train/samples.csv",
    "artifacts/audits/stage2/train/split_fold2_seeded_random_20260924.csv",
    "artifacts/audits/stage2/train/label_conflicts.csv",
    "artifacts/experiments/20260925-stage2-fold2-c5-h1-336-seed20260921/best.pt",
    "artifacts/weights/ViT-B-32.pt",
    CONTROL,
    CANDIDATE,
)
OUTPUTS = (
    "artifacts/experiments/20260929-stage2-p11-fold2-sce-control",
    "artifacts/experiments/20260929-stage2-p11-fold2-nce-rce-candidate",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_configs() -> None:
    first = yaml.safe_load((ROOT / CONTROL).read_text(encoding="utf-8"))["experiment"]
    second = yaml.safe_load((ROOT / CANDIDATE).read_text(encoding="utf-8"))["experiment"]
    changed = {key for key in first if first[key] != second[key]}
    if changed != {"run_id", "output_dir", "base_loss"}:
        raise RuntimeError(f"P11 pair has unintended config differences: {sorted(changed)}")
    for config in (first, second):
        if config["execution_enabled"] is not True:
            raise RuntimeError("P11 configs must be explicitly enabled after user approval")
        if config["training"]["epochs"] != 8 or config["training"]["seed"] != 20260921:
            raise RuntimeError("P11 training schedule or seed drifted")
        if config["head_checkpoint"].endswith("final.pt"):
            raise RuntimeError("fixed-split experiments must use fold-specific H1 best.pt")
    if first["base_loss"] != {"kind": "sce", "alpha": 0.1, "beta": 1.0, "log_zero": -4.0}:
        raise RuntimeError("P11 control is not the frozen SCE baseline")
    if second["base_loss"] != {"kind": "nce_rce", "alpha": 750.0, "beta": 0.1, "log_zero": -4.0}:
        raise RuntimeError("P11 candidate loss does not match preregistration")


def main() -> int:
    if QUEUE_DIR.exists():
        raise FileExistsError(f"P11 queue already exists: {QUEUE_DIR}; refusing implicit restart")
    if any((ROOT / output).exists() for output in OUTPUTS):
        raise FileExistsError("P11 output path already exists; preserve it and inspect manually")
    if read_training_lock(ROOT / "artifacts/queues/training.lock.json") is not None:
        raise RuntimeError("training lock is not free")
    if probe_running_trainers():
        raise RuntimeError("a supported training process is already running")
    if not torch.cuda.is_available():
        raise RuntimeError("P11 requires the local CUDA GPU")
    free_gpu_bytes, total_gpu_bytes = torch.cuda.mem_get_info(0)
    if free_gpu_bytes < 5 * 1024**3:
        raise RuntimeError(f"P11 needs >=5 GiB currently free VRAM; got {free_gpu_bytes/1024**3:.2f} GiB")
    disk_free = shutil.disk_usage(ROOT).free
    if disk_free < 12 * 1024**3:
        raise RuntimeError(f"P11 needs >=12 GiB free D: space; got {disk_free/1024**3:.2f} GiB")
    _validate_configs()
    frozen_sha = {relative: _sha256(ROOT / relative) for relative in SOURCE_INPUTS}

    started = datetime.now(ZoneInfo("Asia/Shanghai"))
    QUEUE_DIR.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": "aictrack1.p11_approval_receipt.v1",
        "queue_name": QUEUE_NAME,
        "scope": "local RTX4060; Stage2 fold2 train/validation only; matched 8-epoch SCE control then NCE+RCE candidate",
        "approval": "User authorized continued local-only training with system resource scheduling until next eligible package",
        "started_at": started.isoformat(),
        "expected_pair_wall_hours": [2.5, 4.0],
        "hard_budget_hours": 6.0,
        "hard_deadline": (started + timedelta(hours=6)).isoformat(),
        "expected_disk_delta_gib": [2.0, 4.0],
        "disk_free_bytes_before": disk_free,
        "gpu_free_bytes_before": free_gpu_bytes,
        "gpu_total_bytes": total_gpu_bytes,
        "test_access": False,
        "source_input_sha256": frozen_sha,
    }
    (QUEUE_DIR / "approval_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    items = [
        QueueItem(
            tag="sce_control",
            config=CONTROL,
            output_dir=OUTPUTS[0],
            purpose="matched fold2 local SCE control, identical PEFT/AC/seed/split/hardware",
        ),
        QueueItem(
            tag="nce_rce_candidate",
            config=CANDIDATE,
            output_dir=OUTPUTS[1],
            purpose="single-variable NCE+RCE loss candidate",
        ),
    ]

    def frozen_launcher(command, cwd, output, on_start):
        actual = {relative: _sha256(ROOT / relative) for relative in SOURCE_INPUTS}
        if actual != frozen_sha:
            raise RuntimeError("frozen P11 source/config/input SHA changed before child launch")
        from aictrack1.engine.training_queue import launch_and_wait

        return launch_and_wait(command, cwd, output, on_start)

    return run_serial_queue(
        root=ROOT,
        python=Path(sys.executable),
        logs=QUEUE_DIR,
        queue_name=QUEUE_NAME,
        items=items,
        launcher=frozen_launcher,
    )


if __name__ == "__main__":
    raise SystemExit(main())
