"""Bounded train-only four-view comparison of P17 EMA and its own raw trajectory."""

from __future__ import annotations

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
from aictrack1.data.manifest import sha256_file  # noqa: E402
from aictrack1.engine.training_queue import TrainingLease, probe_running_trainers  # noqa: E402

EXPECTED_ROOT = Path("/root/autodl-tmp/AICtrack1_p17_diagnostics_20261001")
ORIGINAL_LOCK = Path("/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json")
RUN_NAME = "20261001-stage2-p17-fold2-sce-ac-ema0999-8ep-4090-w16-v4"
RUN = ROOT / "artifacts/experiments" / RUN_NAME
CONFIG = ROOT / "configs/experiment/stage2_p17_fold2_sce_ac_ema0999_8ep_4090.yaml"
QUEUE = ROOT / "artifacts/queues/20261001-p17-ema-raw-four-view-v2"
REPORT = ROOT / "artifacts/reports/20261001-p17-fold2-ema-vs-raw-grouped.json"
HARD_SECONDS = 40 * 60
MAX_ADDED = 1024**3


def _run(name: str, command: list[str], deadline: float, free_start: int) -> None:
    with (QUEUE / f"{name}.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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
                raise RuntimeError(f"P17 diagnostic cutoff during {name}: remaining={remaining:.1f}, free={free}")
            time.sleep(min(5, remaining))
        if child.returncode != 0:
            raise RuntimeError(f"P17 diagnostic {name} failed with exit code {child.returncode}")


def main() -> int:
    if ROOT != EXPECTED_ROOT or (ROOT / "DataSetStage2/test").exists():
        raise RuntimeError("P17 diagnostics require the isolated train-only workspace")
    if QUEUE.exists() or REPORT.exists() or probe_running_trainers():
        raise RuntimeError("P17 diagnostic output exists or another trainer is active")
    metrics = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    if len(metrics["history"]) != 8 or metrics["history"][-1]["epoch"] != 8:
        raise RuntimeError("P17 training did not complete all eight epochs")
    if metrics["selected_checkpoint"] != "best.pt":
        raise RuntimeError("P17 selected checkpoint is not best.pt")
    free_start = shutil.disk_usage(ROOT).free
    if free_start < 7 * 1024**3:
        raise RuntimeError("P17 diagnostic disk headroom is insufficient")
    checksums = {name: sha256_file(RUN / name) for name in ("best.pt", "raw_best.pt", "last.pt")}
    if checksums["best.pt"] != metrics["selected_checkpoint_sha256"] or checksums["last.pt"] != metrics["last_checkpoint_sha256"]:
        raise RuntimeError("P17 selected or last checkpoint SHA mismatch")
    required_last = {
        "epoch", "trainable_state", "optimizer", "scheduler", "scaler", "python_rng_state",
        "numpy_rng_state", "torch_rng_state", "cuda_rng_state_all", "confidence_weighting", "trainable_ema",
        "raw_best_state", "raw_best_top1",
    }
    for name in checksums:
        state = torch.load(RUN / name, map_location="cpu", weights_only=True)
        if name == "last.pt" and (not required_last.issubset(state) or state["epoch"] != 8):
            raise RuntimeError("P17 final checkpoint is not complete and safely resumable")
        if name == "raw_best.pt" and state.get("selection_rule") != "raw best fixed-split validation Top-1; diagnostic only":
            raise RuntimeError("P17 raw diagnostic checkpoint has incorrect provenance")
        del state
    QUEUE.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + HARD_SECONDS
    (QUEUE / "source_receipt.json").write_text(json.dumps({
        "scope": "Stage2 train-derived fold2 29,737 rows; EMA versus own raw trajectory",
        "checkpoint_sha256": checksums,
        "config_sha256": sha256_file(CONFIG),
        "diagnose_script_sha256": sha256_file(ROOT / "scripts/diagnose_stage2_tta.py"),
        "audit_script_sha256": sha256_file(ROOT / "scripts/audit_stage2_paired_tta_gain.py"),
        "safe_loader_sha256": sha256_file(ROOT / "src/aictrack1/models/single_model.py"),
        "hard_seconds": HARD_SECONDS,
        "max_added_bytes": MAX_ADDED,
        "started_unix": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    lease = TrainingLease.acquire(
        ORIGINAL_LOCK, owner="20261001-p17-ema-raw-four-view",
        config=str(CONFIG), output_dir=str(QUEUE), command=tuple(sys.argv),
    )
    try:
        for role, name in (("ema", "best.pt"), ("raw", "raw_best.pt")):
            output = ROOT / "artifacts/diagnostics" / f"20261001-p17-fold2-{role}-384-four-view"
            if output.exists():
                raise FileExistsError(output)
            remaining = int(deadline - time.monotonic())
            if remaining < 120:
                raise RuntimeError("P17 diagnostic budget insufficient for next checkpoint")
            command = [
                sys.executable, "scripts/diagnose_stage2_tta.py",
                "--experiment-config", str(CONFIG), "--checkpoint", str(RUN / name),
                "--expected-checkpoint-sha256", checksums[name],
                "--output-dir", str(output), "--batch-size", "64", "--workers", "16",
                "--four-view-scale384", "--hard-seconds", str(remaining - 30),
            ]
            if role == "raw":
                command.append("--raw-ema-diagnostic")
            _run(role, command, deadline, free_start)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            if report["evaluated_count"] != 29737 or report["checkpoint_sha256"] != checksums[name]:
                raise RuntimeError(f"P17 {role} diagnostic report is incomplete")
        baseline = ROOT / "artifacts/diagnostics/20261001-p17-fold2-raw-384-four-view/paired_predictions.csv"
        candidate = ROOT / "artifacts/diagnostics/20261001-p17-fold2-ema-384-four-view/paired_predictions.csv"
        _run("grouped_audit", [
            sys.executable, "scripts/audit_stage2_paired_tta_gain.py",
            "--baseline-predictions", str(baseline), "--candidate-predictions", str(candidate),
            "--baseline-column", "four_view_prediction", "--candidate-column", "four_view_prediction",
            "--seed", "20260925", "--replicates", "500", "--output", str(REPORT),
        ], deadline, free_start)
        result = {"status": "complete", "report_sha256": sha256_file(REPORT), "checkpoint_sha256": checksums}
        (QUEUE / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return 0
    finally:
        lease.release()


if __name__ == "__main__":
    raise SystemExit(main())
