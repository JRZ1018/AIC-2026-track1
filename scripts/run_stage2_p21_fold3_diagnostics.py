"""Bounded, serial P21 fold3 four-view validation after the completed pair."""
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
from aictrack1.engine.training_queue import TrainingLease, probe_running_trainers

EXPECTED_ROOT = Path("/root/autodl-tmp/AICtrack1_p21_fold3_pair_20261001")
LOCK = Path("/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json")
TRAIN_QUEUE = ROOT / "artifacts/queues/20261001-p21-fold3-4090-14-vs20-matched"
QUEUE = ROOT / "artifacts/queues/20261001-p21-fold3-14-vs20-fourview384"
GATE = ROOT / "artifacts/reports/20261001-p21-fold3-20-vs14-matched-gate.json"
HARD_SECONDS = 3600
MAX_ADDED = 2 * 1024**3
REQUIRED_STATE = {"trainable_state", "optimizer", "scheduler", "scaler", "epoch", "history",
                  "best_state", "python_rng_state", "numpy_rng_state", "torch_rng_state",
                  "cuda_rng_state_all", "confidence_weighting"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config(n: int) -> Path:
    return ROOT / f"configs/experiment/stage2_p21_fold3_sce_ac_{n}ep_4090_matched.yaml"


def run_dir(n: int) -> Path:
    return ROOT / f"artifacts/experiments/20261001-stage2-p21-fold3-sce-ac-{n}ep-4090-matched"


def diagnostic(n: int) -> Path:
    return ROOT / f"artifacts/diagnostics/20261001-p21-fold3-{n}ep-fourview384-matched"


def validate_history(n: int, metrics: dict, check: dict) -> None:
    history = metrics["history"]
    if [row["epoch"] for row in history] != list(range(1, n + 1)):
        raise ValueError("P21 fold3 must have every complete epoch in order")
    if any(row["confidence_weighting"]["next_effective_quality"]["decision"] not in {"L0", "L1"}
           for row in history):
        raise ValueError("P21 fold3 AC failed the non-L2 gate")
    if (check.get("complete_epochs") != n
            or check.get("best_sha256") != metrics["selected_checkpoint_sha256"]
            or check.get("last_sha256") != metrics["last_checkpoint_sha256"]):
        raise ValueError("P21 fold3 training receipt/checkpoint selection mismatch")


def training_evidence(*, safe_load: bool) -> dict:
    result = json.loads((TRAIN_QUEUE / "queue_result.json").read_text(encoding="utf-8"))
    if result.get("exit_code") != 0 or set(result.get("checks", {})) != {"14", "20"}:
        raise ValueError("P21 fold3 pair has not completed successfully")
    for n in (14, 20):
        run = run_dir(n)
        check = result["checks"][str(n)]
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        validate_history(n, metrics, check)
        for role in ("best", "last"):
            if sha(run / f"{role}.pt") != check[f"{role}_sha256"]:
                raise ValueError(f"P21 fold3 {n}-epoch {role} SHA mismatch")
        if safe_load:
            state = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
            if not REQUIRED_STATE.issubset(state) or state["epoch"] != n or len(state["history"]) != n:
                raise ValueError("P21 fold3 last is not safely resumable")
            del state
            state = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
            if state.get("training_scope") != "fixed_split_validation":
                raise ValueError("P21 fold3 best is not train-derived validation selected")
            del state
    return result


def stop_child(child: subprocess.Popen) -> None:
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=20)


def main() -> int:
    if sys.platform != "linux" or ROOT != EXPECTED_ROOT or (ROOT / "DataSetStage2/test").exists():
        raise RuntimeError("exact P21 fold3 train-only root required")
    if (LOCK.exists() or probe_running_trainers() or QUEUE.exists() or GATE.exists()
            or any(diagnostic(n).exists() for n in (14, 20))):
        raise RuntimeError("existing trainer, lock, diagnostic, queue, or gate")
    started = time.time()
    deadline = time.monotonic() + HARD_SECONDS
    free_start = shutil.disk_usage(ROOT).free
    if free_start < 7 * 1024**3:
        raise RuntimeError("P21 fold3 diagnostic disk headroom insufficient")
    frozen = json.loads((ROOT / "p21_fold3_input_sha256.json").read_text(encoding="utf-8"))
    if any(sha(ROOT / p) != h for p, h in frozen.items()):
        raise RuntimeError("P21 fold3 frozen training source/input changed")
    evidence = training_evidence(safe_load=True)
    QUEUE.mkdir(parents=True, exist_ok=False)
    (QUEUE / "source_receipt.json").write_text(json.dumps({
        "scope": "Stage2 train-derived fold3, two 29,737-row fixed384 four-view diagnostics; no test",
        "training_result_sha256": sha(TRAIN_QUEUE / "queue_result.json"),
        "frozen_sha256": frozen, "checks": evidence["checks"],
        "new_entry_sha256": {p: sha(ROOT / p) for p in (
            "scripts/run_stage2_p21_fold3_diagnostics.py", "scripts/audit_stage2_p21_fold3_schedule_pair.py")},
        "started_unix": started, "hard_seconds": HARD_SECONDS, "max_added_bytes": MAX_ADDED,
    }, indent=2) + "\n", encoding="utf-8")
    lease = TrainingLease.acquire(LOCK, owner="20261001-p21-fold3-14-vs20-fourview384",
                                  config=str(config(14)), output_dir=str(QUEUE), command=tuple(sys.argv))
    try:
        commands = []
        for n in (14, 20):
            commands.append((f"{n}ep", [sys.executable, "scripts/diagnose_stage2_tta.py",
                "--experiment-config", str(config(n)), "--checkpoint", str(run_dir(n) / "best.pt"),
                "--expected-checkpoint-sha256", evidence["checks"][str(n)]["best_sha256"],
                "--output-dir", str(diagnostic(n)), "--batch-size", "64", "--workers", "16",
                "--four-view-scale384"]))
        commands.append(("gate", [sys.executable, "scripts/audit_stage2_p21_fold3_schedule_pair.py",
            "--control-dir", str(diagnostic(14)), "--candidate-dir", str(diagnostic(20)),
            "--output", str(GATE)]))
        for name, command in commands:
            remaining = deadline - time.monotonic()
            if remaining < 30:
                raise RuntimeError("P21 fold3 diagnostic budget exhausted")
            if name != "gate":
                command += ["--hard-seconds", str(int(remaining) - 20)]
            with (QUEUE / f"{name}.log").open("w", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                try:
                    lease.attach_trainer(child.pid)
                    while child.poll() is None:
                        free = shutil.disk_usage(ROOT).free
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 or free < 5 * 1024**3 or free_start - free > MAX_ADDED:
                            raise RuntimeError("P21 fold3 diagnostic hard/resource cutoff")
                        time.sleep(min(5, remaining))
                    if child.returncode != 0:
                        raise RuntimeError(f"P21 fold3 {name} diagnostic failed: {child.returncode}")
                finally:
                    stop_child(child)
        gate = json.loads(GATE.read_text(encoding="utf-8"))
        result = {"status": "complete", "gate_passed": gate["gate_passed"], "gate_sha256": sha(GATE),
                  "diagnostics": {str(n): {"report_sha256": sha(diagnostic(n) / "report.json"),
                    "predictions_sha256": sha(diagnostic(n) / "paired_predictions.csv")} for n in (14, 20)}}
        (QUEUE / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return 0
    finally:
        lease.release()


if __name__ == "__main__":
    raise SystemExit(main())
