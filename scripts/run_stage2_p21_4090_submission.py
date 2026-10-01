"""One frozen P21 fold-checkpoint test inference for a user-requested exploratory package.

This deliberately bypasses the P21 matched-schedule advancement gate at the
user's explicit request. It does not claim a validated improvement or tune on
test/online feedback. Only the pre-existing P7 inference protocol is used.
"""

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

EXPECTED_ROOT = Path("/root/autodl-tmp/AICtrack1_p21_20ep_20261001")
ORIGINAL_LOCK = Path("/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json")
RUN = ROOT / "artifacts/experiments/20261001-stage2-p21-fold2-sce-ac-20ep-4090-screen"
CONFIG = ROOT / "configs/experiment/stage2_p21_fold2_sce_ac_20ep_4090_screen.yaml"
DIAGNOSTIC = ROOT / "artifacts/diagnostics/20261001-p21-fold2-20ep-fourview384"
QUEUE = ROOT / "artifacts/queues/20261001-p21-fold2-20ep-exploratory-submission"
OUTPUT = ROOT / "submissions/20261001_p21_fold2_20ep_fourview_balanced_exploratory"
TEST_MANIFEST = ROOT / "artifacts/audits/stage2/test/samples.csv"
SELECTION_REPORT = ROOT / "artifacts/reports/20260928-p7-balanced-prior-fold2-fold3.json"
EXPECTED_TEST_SHA = "576ff0f69e0a104ca5a0af7f190628074b572102c6bc8daf8cb653fb08054201"
EXPECTED_SELECTION_SHA = "8122c9c96fb80823db9d19edb403d8d147d26a32c43b7b48fe7cca9db311d703"
HARD_SECONDS = 3600
MAX_ADDED = 1024**3


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if sys.platform != "linux" or ROOT != EXPECTED_ROOT:
        raise RuntimeError("P21 submission requires its exact isolated source root")
    if QUEUE.exists() or OUTPUT.exists() or ORIGINAL_LOCK.exists() or probe_running_trainers():
        raise RuntimeError("inference queue/output or another trainer/shared lock exists")
    if not torch.cuda.is_available() or "RTX 4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("RTX4090 is not available for frozen inference")
    if not (ROOT / "DataSetStage2/test").is_dir():
        raise RuntimeError("official Stage2 test is not staged")
    if sha(TEST_MANIFEST) != EXPECTED_TEST_SHA or sha(SELECTION_REPORT) != EXPECTED_SELECTION_SHA:
        raise RuntimeError("test manifest or previously frozen P7 prior report differs")
    metrics = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    result = json.loads((DIAGNOSTIC.parent.parent / "queues/20261001-p21-fold2-4090-fourview384/result.json")
                        .read_text(encoding="utf-8"))
    report = json.loads((DIAGNOSTIC / "report.json").read_text(encoding="utf-8"))
    best = RUN / "best.pt"
    best_sha = sha(best)
    if (len(metrics["history"]) != 20 or metrics["history"][-1]["epoch"] != 20
            or any(row["confidence_weighting"]["next_effective_quality"]["decision"] == "L2"
                   for row in metrics["history"])
            or best_sha != metrics["selected_checkpoint_sha256"]
            or result["status"] != "complete" or result["best_sha256"] != best_sha
            or report["checkpoint_sha256"] != best_sha or report["evaluated_count"] != 29737):
        raise RuntimeError("P21 model/train-derived diagnostic provenance failed")
    checkpoint = torch.load(best, map_location="cpu", weights_only=True)
    if checkpoint.get("competition_stage") != "stage2" or checkpoint.get("num_classes") != 750:
        raise RuntimeError("P21 checkpoint is not a Stage2 750-class single model")
    del checkpoint
    free_start = shutil.disk_usage(ROOT).free
    if free_start < 7 * 1024**3:
        raise RuntimeError("insufficient inference disk headroom")
    QUEUE.mkdir(parents=True, exist_ok=False)
    receipt = {"scope": "one official Stage2 test inference; no tuning or online submission",
               "authorization": "2026-10-01 user requested immediate exploratory P21 package",
               "qualification": "P21 matched 14-epoch control not run; online gain is unproven",
               "checkpoint_sha256": best_sha, "config_sha256": sha(CONFIG),
               "train_derived_diagnostic_sha256": sha(DIAGNOSTIC / "report.json"),
               "test_manifest_sha256": EXPECTED_TEST_SHA,
               "p7_balanced_selection_sha256": EXPECTED_SELECTION_SHA,
               "inference_source_sha256": sha(ROOT / "scripts/infer_and_build_submission.py"),
               "hard_seconds": HARD_SECONDS, "max_added_bytes": MAX_ADDED,
               "started_unix": time.time()}
    (QUEUE / "source_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    lease = TrainingLease.acquire(ORIGINAL_LOCK, owner="20261001-p21-exploratory-test-inference",
                                  config=str(CONFIG), output_dir=str(OUTPUT), command=tuple(sys.argv))
    try:
        command = [sys.executable, "scripts/infer_and_build_submission.py",
                   "--experiment-config", str(CONFIG), "--checkpoint", str(best),
                   "--test-manifest", str(TEST_MANIFEST), "--output-dir", str(OUTPUT),
                   "--batch-size", "64", "--workers", "16", "--four-view-scale384",
                   "--expected-checkpoint-sha256", best_sha,
                   "--balanced-prior-uniform-soft",
                   "--balanced-prior-temperature-multiplier", "0.5",
                   "--balanced-prior-bias-strength", "0.75",
                   "--balanced-prior-selection-report", str(SELECTION_REPORT),
                   "--expected-balanced-prior-report-sha256", EXPECTED_SELECTION_SHA]
        deadline = time.monotonic() + HARD_SECONDS
        with (QUEUE / "inference.log").open("w", encoding="utf-8") as log:
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
                    raise RuntimeError(f"P21 inference hard/resource cutoff: free={free}")
                time.sleep(min(5, remaining))
            if child.returncode != 0:
                raise RuntimeError(f"P21 inference failed with exit code {child.returncode}")
        manifest = json.loads((OUTPUT / "submission_manifest.json").read_text(encoding="utf-8"))
        if manifest["record_count"] != 37444:
            raise RuntimeError("P21 package does not have 37,444 rows")
        result = {"status": "package_generated_pending_independent_audit",
                  "checkpoint_sha256": best_sha, "csv_sha256": sha(OUTPUT / "pred_results.csv"),
                  "zip_sha256": sha(OUTPUT / "pred_results.zip"),
                  "manifest_sha256": sha(OUTPUT / "submission_manifest.json")}
        (QUEUE / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return 0
    finally:
        lease.release()


if __name__ == "__main__":
    raise SystemExit(main())
