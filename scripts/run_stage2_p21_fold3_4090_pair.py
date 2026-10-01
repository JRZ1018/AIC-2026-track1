"""Bounded P21 fold3 same-source 14/20 epoch confirmation; no test."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aictrack1.engine.training_queue import QueueItem, probe_running_trainers, run_serial_queue
from run_stage2_p21_fold2_4090_20ep import sha

EXPECTED_ROOT = Path("/root/autodl-tmp/AICtrack1_p21_fold3_pair_20261001")
LOCK = Path("/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json")
NAME = "20261001-p21-fold3-4090-14-vs20-matched"
QUEUE = ROOT / "artifacts/queues" / NAME
GATE = ROOT / "artifacts/reports/20261001-p21-fold2-20-vs14-matched-gate.json"
GATE_SHA = "c630d2f8b54040d477eb5e1dac2c6f8a427e691bf1957b9a5ba819f95a6f9873"
HARD_SECONDS = 5 * 3600
MAX_ADDED = 6 * 1024**3


def config(n: int) -> str:
    return f"configs/experiment/stage2_p21_fold3_sce_ac_{n}ep_4090_matched.yaml"


def output(n: int) -> str:
    return f"artifacts/experiments/20261001-stage2-p21-fold3-sce-ac-{n}ep-4090-matched"


def preflight() -> dict:
    if sys.platform != "linux" or ROOT != EXPECTED_ROOT or (ROOT / "DataSetStage2/test").exists():
        raise RuntimeError("exact P21 fold3 train-only root required")
    if LOCK.exists() or probe_running_trainers() or QUEUE.exists() or any((ROOT/output(n)).exists() for n in (14, 20)):
        raise RuntimeError("existing trainer, lock, queue, or output")
    if sha(GATE) != GATE_SHA or json.loads(GATE.read_text())["gate_passed"] is not True:
        raise RuntimeError("fold2 pre-registered gate not passed")
    if not torch.cuda.is_available() or "RTX 4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("approved RTX4090 unavailable")
    free = shutil.disk_usage(ROOT).free
    gpu_free, _ = torch.cuda.mem_get_info(0)
    if free < 12*1024**3 or gpu_free < 20*1024**3:
        raise RuntimeError("resource headroom insufficient")
    frozen = json.loads((ROOT / "p21_fold3_input_sha256.json").read_text())
    drift = [p for p, h in frozen.items() if not (ROOT/p).is_file() or sha(ROOT/p) != h]
    if drift:
        raise RuntimeError(f"frozen source/input drift: {drift}")
    baseline = yaml.safe_load((ROOT / "configs/experiment/stage2_p21_fold2_sce_ac_20ep_4090_screen.yaml").read_text())["experiment"]
    for n in (14, 20):
        actual = yaml.safe_load((ROOT/config(n)).read_text())["experiment"]
        expected = dict(baseline)
        expected.update(run_id=Path(output(n)).name, output_dir="../../"+output(n),
            dataset_config="../data/stage2_fold3_336.yaml",
            split="../../artifacts/audits/stage2/train/split_fold3_seeded_random_20260925.csv",
            head_checkpoint="../../artifacts/experiments/20260925-stage2-fold3-c5-h1-336-seed20260921/best.pt",
            expected_initial_top1=0.6629787683486938,
            training={**baseline["training"], "epochs": n})
        if actual != expected:
            raise RuntimeError("fold3 configuration differs beyond fold/schedule/identity")
    return {"scope": "Stage2 train-derived fold3 matched 14/20; no test",
            "frozen_sha256": frozen, "fold2_gate_sha256": GATE_SHA,
            "disk_free": free, "gpu_free": gpu_free, "hard_seconds": HARD_SECONDS,
            "max_added_bytes": MAX_ADDED}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    receipt = preflight()
    if not args.run:
        print(json.dumps({k:v for k,v in receipt.items() if k != "frozen_sha256"}, indent=2))
        return 0
    deadline = time.time() + HARD_SECONDS
    QUEUE.mkdir(parents=True, exist_ok=False)
    receipt.update(started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                   deadline=datetime.fromtimestamp(deadline, ZoneInfo("Asia/Shanghai")).isoformat())
    (QUEUE/"source_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")

    def launch(command, cwd, log, on_start):
        if any(sha(ROOT/p) != h for p,h in receipt["frozen_sha256"].items()):
            raise RuntimeError("frozen source/input changed before training")
        child = subprocess.Popen(list(command), cwd=cwd, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            on_start(child.pid)
            while child.poll() is None:
                free = shutil.disk_usage(ROOT).free
                if time.time() >= deadline or free < 5*1024**3 or receipt["disk_free"]-free > MAX_ADDED:
                    log.write("P21 fold3 hard/resource cutoff\n"); log.flush()
                    return 124
                time.sleep(5)
            return child.returncode
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=30)

    code = run_serial_queue(root=ROOT, python=Path(sys.executable), logs=QUEUE,
        queue_name=NAME, lock_path=LOCK,
        items=[QueueItem(f"matched_{n}ep", config(n), output(n), f"P21 fold3 {n} epochs") for n in (14,20)],
        launcher=launch)
    checks = {}
    if code == 0:
        for n in (14,20):
            run = ROOT/output(n)
            metrics = json.loads((run/"metrics.json").read_text())
            if len(metrics["history"]) != n or metrics["history"][-1]["epoch"] != n:
                raise RuntimeError("incomplete fold3 epochs")
            if any(r["confidence_weighting"]["next_effective_quality"]["decision"] == "L2" for r in metrics["history"]):
                raise RuntimeError("fold3 AC L2")
            best_sha, last_sha = sha(run/"best.pt"), sha(run/"last.pt")
            if best_sha != metrics["selected_checkpoint_sha256"] or last_sha != metrics["last_checkpoint_sha256"]:
                raise RuntimeError("fold3 checkpoint SHA mismatch")
            state = torch.load(run/"last.pt", map_location="cpu", weights_only=True)
            required = {"trainable_state", "optimizer", "scheduler", "scaler", "epoch", "history", "best_state",
                        "python_rng_state", "numpy_rng_state", "torch_rng_state", "cuda_rng_state_all", "confidence_weighting"}
            if not required.issubset(state) or state["epoch"] != n:
                raise RuntimeError("fold3 last not safely resumable")
            del state
            torch.load(run/"best.pt", map_location="cpu", weights_only=True)
            checks[str(n)] = {"best_sha256":best_sha, "last_sha256":last_sha, "complete_epochs":n}
    (QUEUE/"queue_result.json").write_text(json.dumps({"exit_code":code, "checks":checks,
        "finished_at":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}, indent=2)+"\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
