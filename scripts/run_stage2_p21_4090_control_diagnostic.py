"""Bounded train-only four-view diagnostic for the P21 matched 14-epoch control."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aictrack1.engine.training_queue import TrainingLease, probe_running_trainers  # noqa: E402

EXPECTED_ROOT = Path("/root/autodl-tmp/AICtrack1_p21_fold2_control_20261001")
LOCK = Path("/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json")
RUN = ROOT / "artifacts/experiments/20261001-stage2-p21-fold2-sce-ac-14ep-4090-matched"
CONFIG = ROOT / "configs/experiment/stage2_p21_fold2_sce_ac_14ep_4090_matched.yaml"
TRAIN_QUEUE = ROOT / "artifacts/queues/20261001-p21-fold2-4090-14ep-matched"
QUEUE = ROOT / "artifacts/queues/20261001-p21-fold2-4090-14ep-fourview384"
DIAGNOSTIC = ROOT / "artifacts/diagnostics/20261001-p21-fold2-14ep-fourview384-matched"
HARD_SECONDS = 40 * 60
MAX_ADDED = 1024**3


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if sys.platform != "linux" or ROOT != EXPECTED_ROOT or (ROOT / "DataSetStage2/test").exists():
        raise RuntimeError("P21 control diagnostics require the exact train-only root")
    if QUEUE.exists() or DIAGNOSTIC.exists() or LOCK.exists() or probe_running_trainers():
        raise RuntimeError("P21 control diagnostic output, trainer, or shared lock exists")
    training_result = json.loads((TRAIN_QUEUE / "queue_result.json").read_text(encoding="utf-8"))
    if training_result.get("exit_code") != 0:
        raise RuntimeError("P21 matched control training did not complete normally")
    metrics = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    history = metrics["history"]
    if len(history) != 14 or history[-1]["epoch"] != 14:
        raise RuntimeError("P21 matched control fourteen epochs are incomplete")
    if any(row["confidence_weighting"]["next_effective_quality"]["decision"] == "L2"
           for row in history):
        raise RuntimeError("P21 matched control AC quality fell to L2")
    best, last = RUN / "best.pt", RUN / "last.pt"
    best_sha, last_sha = sha(best), sha(last)
    if best_sha != metrics["selected_checkpoint_sha256"] or last_sha != metrics["last_checkpoint_sha256"]:
        raise RuntimeError("P21 matched control checkpoint SHA mismatch")
    final_state = torch.load(last, map_location="cpu", weights_only=True)
    required = {"trainable_state", "optimizer", "scheduler", "scaler", "epoch", "history",
                "best_state", "python_rng_state", "numpy_rng_state", "torch_rng_state",
                "cuda_rng_state_all", "confidence_weighting"}
    if not required.issubset(final_state) or final_state["epoch"] != 14:
        raise RuntimeError("P21 matched control last is not safely resumable")
    torch.load(best, map_location="cpu", weights_only=True)
    del final_state
    free_start = shutil.disk_usage(ROOT).free
    if free_start < 7 * 1024**3:
        raise RuntimeError("P21 matched control diagnostic disk headroom is insufficient")
    QUEUE.mkdir(parents=True, exist_ok=False)
    (QUEUE / "source_receipt.json").write_text(json.dumps({
        "scope": "Stage2 train-derived fold2 29,737 rows; fixed 384 four views",
        "checkpoint_sha256": best_sha, "last_sha256": last_sha,
        "config_sha256": sha(CONFIG), "diagnose_script_sha256": sha(ROOT / "scripts/diagnose_stage2_tta.py"),
        "hard_seconds": HARD_SECONDS, "max_added_bytes": MAX_ADDED, "started_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    lease = TrainingLease.acquire(LOCK, owner="20261001-p21-fold2-14ep-fourview384",
                                  config=str(CONFIG), output_dir=str(DIAGNOSTIC), command=tuple(sys.argv))
    try:
        deadline = time.monotonic() + HARD_SECONDS
        command = [sys.executable, "scripts/diagnose_stage2_tta.py",
                   "--experiment-config", str(CONFIG), "--checkpoint", str(best),
                   "--expected-checkpoint-sha256", best_sha,
                   "--output-dir", str(DIAGNOSTIC), "--batch-size", "64", "--workers", "16",
                   "--four-view-scale384", "--hard-seconds", str(HARD_SECONDS - 30)]
        with (QUEUE / "diagnostic.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            while child.poll() is None:
                remaining = deadline - time.monotonic()
                free = shutil.disk_usage(ROOT).free
                if remaining <= 0 or free_start - free > MAX_ADDED or free < 5 * 1024**3:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=20)
                    raise RuntimeError(f"P21 matched control diagnostic hard/resource cutoff: free={free}")
                time.sleep(min(5, remaining))
            if child.returncode != 0:
                raise RuntimeError(f"P21 matched control four-view diagnostic exited {child.returncode}")
        report = json.loads((DIAGNOSTIC / "report.json").read_text(encoding="utf-8"))
        if report["evaluated_count"] != 29737 or report["checkpoint_sha256"] != best_sha:
            raise RuntimeError("P21 matched control diagnostic report incomplete")
        result = {"status": "complete", "report_sha256": sha(DIAGNOSTIC / "report.json"),
                  "predictions_sha256": sha(DIAGNOSTIC / "paired_predictions.csv"),
                  "best_sha256": best_sha, "last_sha256": last_sha}
        (QUEUE / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return 0
    finally:
        lease.release()


if __name__ == "__main__":
    raise SystemExit(main())
