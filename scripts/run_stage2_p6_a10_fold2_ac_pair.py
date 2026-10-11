"""One approved A10 fold2 SCE/AC ablation under the shared training lease."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import QueueItem, run_parallel_queue


QUEUE_NAME = "20260928-p6-a10-fold2-sce-ac-pair"
LOGS = ROOT / "artifacts/queues" / QUEUE_NAME
CONTROL = "configs/experiment/stage2_p6_sce_fold2_a10_ac_on_control.yaml"
CANDIDATE = "configs/experiment/stage2_p6_sce_fold2_a10_ac_off_candidate.yaml"
EXPECTED_SHA256 = {
    "scripts/train_peft.py": "08b42a7aea3778d0c002152d2da893f4db0b2feeeac9c1ebee68fd6ed1f787a1",
    "src/aictrack1/engine/training_queue.py": "f3f2121e9a3e5dd63e4953d2732cd9f1f983d61cd8214e1c8b675e568ca3bfb7",
    "src/aictrack1/methods/sce.py": "5eb2da417cc47bc8f484d9cda03f453b3f1a079e9a8635c60ba6261ccb3687a9",
    "artifacts/audits/stage2/train/samples.csv": "1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d",
    "artifacts/audits/stage2/train/split_fold2_seeded_random_20260924.csv": "1e5eb7e9adbc234ba933bf76c99896602b8bdce5557879f90224ae2bb4cf2a50",
    "artifacts/experiments/20260925-stage2-fold2-c5-h1-336-seed20260921/best.pt": "83380620f06af442ddbe4e55f048d6519d03ce95e672ab3bc51469904b803500",
    "artifacts/weights/ViT-B-32.pt": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
    CONTROL: "f6997daa8b2cff765c88d3a16c15a5c34ab4c4efe0d759069383b27199cb8b47",
    CANDIDATE: "b9eddb478620f3044ae2bb0a9cc0cb373fb0284932dbaaa3b54e4167965f7926",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_configs() -> None:
    control = yaml.safe_load((ROOT / CONTROL).read_text(encoding="utf-8"))["experiment"]
    candidate = yaml.safe_load((ROOT / CANDIDATE).read_text(encoding="utf-8"))["experiment"]
    if not control["execution_enabled"] or not candidate["execution_enabled"]:
        raise RuntimeError("P6 configs must be explicitly enabled after approval")
    changed = {key for key in control if control[key] != candidate[key]}
    if changed != {"run_id", "output_dir", "confidence_weighting"}:
        raise RuntimeError(f"unexpected P6 config differences: {changed}")
    weighting_changed = {
        key
        for key in control["confidence_weighting"]
        if control["confidence_weighting"][key] != candidate["confidence_weighting"][key]
    }
    if weighting_changed != {"enabled"}:
        raise RuntimeError(f"unexpected AC config differences: {weighting_changed}")
    if control["confidence_weighting"]["enabled"] is not True:
        raise RuntimeError("control must enable AC")
    if candidate["confidence_weighting"]["enabled"] is not False:
        raise RuntimeError("candidate must disable AC")


def main() -> int:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("P6 pair is approved only on the A10 Linux host")
    if LOGS.exists():
        raise RuntimeError(f"P6 queue directory already exists: {LOGS}; no implicit restart")
    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes < 12 * 1024**3:
        raise RuntimeError("P6 requires at least 12 GiB free before launch")
    required_ld_paths = (
        "/usr/local/cuda-12.8/extras/CUPTI/lib64",
        "/usr/local/cuda-12.8/targets/x86_64-linux/lib",
        "/usr/local/cuda-12.8/lib64",
    )
    configured_ld_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if any(path not in configured_ld_paths for path in required_ld_paths):
        raise RuntimeError("required CUDA 12.8 runtime paths are missing")
    actual_sha256 = {relative: _sha256(ROOT / relative) for relative in EXPECTED_SHA256}
    if actual_sha256 != EXPECTED_SHA256:
        mismatch = [key for key in EXPECTED_SHA256 if actual_sha256[key] != EXPECTED_SHA256[key]]
        raise RuntimeError(f"frozen P6 input/source SHA mismatch: {mismatch}")
    _check_configs()

    started = datetime.now(ZoneInfo("Asia/Shanghai"))
    LOGS.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": "aictrack1.p6_approval_receipt.v1",
        "queue_name": QUEUE_NAME,
        "approval": "User approved P6 2-4h A10 GPU, hard 6h, 6-10GiB disk on 2026-09-28",
        "started_at": started.isoformat(),
        "hard_deadline": (started + timedelta(hours=6)).isoformat(),
        "hard_deadline_enforcement": "GNU timeout TERM then KILL; launch wrapper must use 6h timeout",
        "expected_sha256": EXPECTED_SHA256,
        "disk_free_bytes_before": free_bytes,
        "test_access": False,
    }
    (LOGS / "approval_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    items = [
        QueueItem(
            tag="sce_ac_on_control",
            config=CONTROL,
            output_dir="artifacts/experiments/20260928-stage2-p6-fold2-a10-sce-ac-on-control",
            purpose="fold2 fixed SCE with AC enabled",
        ),
        QueueItem(
            tag="sce_ac_off_candidate",
            config=CANDIDATE,
            output_dir="artifacts/experiments/20260928-stage2-p6-fold2-a10-sce-ac-off-candidate",
            purpose="fold2 fixed SCE with only AC disabled",
        ),
    ]
    return run_parallel_queue(
        root=ROOT,
        python=Path(sys.executable),
        logs=LOGS,
        queue_name=QUEUE_NAME,
        items=items,
        max_concurrent_jobs=2,
    )


if __name__ == "__main__":
    raise SystemExit(main())
