"""Strict P15 completion audit and bounded train-only fixed384 four-view diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aictrack1.engine.training_queue import TrainingLease, probe_running_trainers, read_training_lock  # noqa: E402

ORIGINAL = Path("/root/autodl-tmp/AICtrack1")
LOCK = ORIGINAL / "artifacts/queues/training.lock.json"
RUN = "20260930-stage2-p15-fold2-sce-ac-visual-ot-8ep-4090"
CONFIG = "configs/experiment/stage2_p15_fold2_sce_ac_visual_ot_8ep_4090.yaml"
QUEUE = ROOT / "artifacts/queues/20260930-p15-fold2-4090-visual-ot-screen"
MODEL = ROOT / "artifacts/experiments" / RUN
OUT = ROOT / "artifacts/diagnostics/20260930-p15-fold2-visual-ot-fourview384"
LOG = ROOT / "artifacts/queues/20260930-p15-fold2-visual-ot-diagnostic.log"
RESULT = ROOT / "artifacts/queues/20260930-p15-fold2-visual-ot-diagnostic-result.json"
CONFIG_SHA = "e2e1b0fc1b5a2cf934d1c8d5c19ce8ef8d71764e762df5b63c0ad4f5a8dbec20"
MANIFEST_SHA = "1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d"
SPLIT_SHA = "1e5eb7e9adbc234ba933bf76c99896602b8bdce5557879f90224ae2bb4cf2a50"
CACHE_SHA = "6f0c3d2811b29b045dad4c6a27c8b01364a0daf2f00c600773ce9ffae12e459a"
LIMIT_SECONDS = 40 * 60
MAX_ADDED = 1024**3
MIN_FREE = 5 * 1024**3


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect() -> dict:
    if sys.platform != "linux" or ROOT != Path("/root/autodl-tmp/AICtrack1_p15_ot01"):
        raise RuntimeError("P15 diagnostic requires exact isolated server project")
    if (ROOT / "DataSetStage2/test").exists():
        raise RuntimeError("P15 diagnostic refuses test-containing workspace")
    if OUT.exists() or LOG.exists() or RESULT.exists():
        raise FileExistsError("P15 diagnostic output already exists; no automatic rerun")
    queue_result = json.loads((QUEUE / "queue_result.json").read_text(encoding="utf-8"))
    receipt = json.loads((QUEUE / "source_receipt.json").read_text(encoding="utf-8"))
    metrics_path = MODEL / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if queue_result.get("exit_code") != 0:
        raise RuntimeError("P15 training did not finish successfully")
    for relative, expected in receipt["input_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"P15 frozen input SHA drift: {relative}")
    if (metrics.get("run_id") != RUN or metrics.get("config_sha256") != CONFIG_SHA
            or metrics.get("manifest_sha256") != MANIFEST_SHA or metrics.get("split_sha256") != SPLIT_SHA
            or metrics.get("training_scope") != "fixed_split_validation" or metrics.get("validation_count") != 29737
            or metrics.get("visual_ot_cache_complete_sha256") != CACHE_SHA
            or metrics.get("selected_checkpoint") != "best.pt"):
        raise RuntimeError("P15 training metrics provenance or selection mismatch")
    ot = metrics.get("visual_ot_diagnostics") or {}
    if (ot.get("source") != "current_stage_train_partition_only" or ot.get("sample_count") != 118958
            or ot.get("class_count") != 750 or ot.get("downweighted_count") != 11896
            or ot.get("temperature") != 0.05 or ot.get("iterations") != 30
            or ot.get("min_weight") != 0.5 or ot.get("low_trust_fraction") != 0.1
            or not math.isfinite(float(ot.get("sinkhorn_marginal_l1", float("nan"))))
            or float(ot["sinkhorn_marginal_l1"]) > 0.05):
        raise RuntimeError("P15 visual OT provenance or numerical diagnostics invalid")
    history = metrics.get("history", [])
    if [row.get("epoch") for row in history] != list(range(1, 9)):
        raise RuntimeError("P15 training did not complete 8 epochs")
    if any(row.get("confidence_weighting", {}).get("next_effective_quality", {}).get("decision") == "L2" for row in history):
        raise RuntimeError("P15 combined AC+OT effective quality reached L2")
    if any(not math.isfinite(float(row.get("train_loss", float("nan")))) or row.get("sampling", {}).get("zero_sample_class_count") != 0 for row in history):
        raise RuntimeError("P15 training history contains nonfinite loss or untrained class")
    best_path, last_path = MODEL / "best.pt", MODEL / "last.pt"
    best_sha, last_sha = sha(best_path), sha(last_path)
    if best_sha != metrics.get("selected_checkpoint_sha256") or last_sha != metrics.get("last_checkpoint_sha256"):
        raise RuntimeError("P15 checkpoint SHA does not match metrics")
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    last = torch.load(last_path, map_location="cpu", weights_only=True)
    if not isinstance(best, dict) or not isinstance(last, dict) or last.get("epoch") != 8:
        raise RuntimeError("P15 checkpoint safe load or full epoch failed")
    required = {"optimizer", "scheduler", "scaler", "python_rng_state", "numpy_rng_state", "torch_rng_state", "cuda_rng_state_all", "confidence_weighting", "history"}
    if not required.issubset(last) or len(last["history"]) != 8:
        raise RuntimeError("P15 last checkpoint missing complete recovery state")
    for checkpoint in (best, last):
        if checkpoint.get("config_sha256") != CONFIG_SHA or not all(torch.isfinite(tensor).all() for tensor in checkpoint["trainable_state"].values()):
            raise RuntimeError("P15 checkpoint config or trainable values invalid")
    if read_training_lock(LOCK) is not None or probe_running_trainers():
        raise RuntimeError("shared lock or trainer active")
    free = shutil.disk_usage(ROOT).free
    if free < 6 * 1024**3:
        raise RuntimeError("P15 diagnostic lacks disk budget")
    return {"best_sha256": best_sha, "last_sha256": last_sha, "metrics_sha256": sha(metrics_path), "disk_free": free, "ot": ot}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    state = inspect()
    if not args.run:
        print(json.dumps({"scope": "Stage2 train-derived fold2 only; no test", **state}, indent=2))
        return 0
    lease = TrainingLease.acquire(LOCK, owner="20260930-p15-fold2-visual-ot-diagnostic", config=str(ROOT / CONFIG), output_dir=str(OUT), command=sys.argv)
    started = time.time()
    code = 1
    try:
        command = [sys.executable, "-u", "scripts/diagnose_stage2_tta.py", "--experiment-config", CONFIG, "--checkpoint", str((MODEL / "best.pt").relative_to(ROOT)), "--expected-checkpoint-sha256", state["best_sha256"], "--output-dir", str(OUT.relative_to(ROOT)), "--batch-size", "64", "--workers", "4", "--four-view-scale384"]
        with LOG.open("w", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                try:
                    code = child.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    free = shutil.disk_usage(ROOT).free
                    if time.time() - started > LIMIT_SECONDS or state["disk_free"] - free > MAX_ADDED or free < MIN_FREE:
                        log.write("P15 diagnostic hard/resource cutoff; retain partial output.\n")
                        log.flush()
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait(timeout=30)
                        code = 124
                        break
        result = {"exit_code": code, "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "elapsed_seconds": time.time() - started, "disk_added_bytes": state["disk_free"] - shutil.disk_usage(ROOT).free, "best_sha256": state["best_sha256"], "report_complete": (OUT / "report.json").is_file(), "predictions_complete": (OUT / "paired_predictions.csv").is_file()}
        RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result))
        return code
    finally:
        lease.release()


if __name__ == "__main__":
    raise SystemExit(main())
