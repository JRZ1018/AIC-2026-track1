"""Explicit one-hour resume of the approved P13 fold2 candidate from epoch 13.

Default is read-only preflight. This is a new queue, not a restart of the
original 5-hour pair. The cumulative disk budget remains 4 GiB.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import yaml

from run_stage2_p13_fold2_4090_schedule_pair import (  # noqa: E402
    EXPECTED_SHA,
    INPUTS as PAIR_INPUTS,
    ROOT,
    free_ram,
    sha256,
    stop_owned_process,
)

sys.path.insert(0, str(ROOT / "src"))
from aictrack1.engine.training_queue import (  # noqa: E402
    DEFAULT_LOCK_RELATIVE_PATH,
    QueueItem,
    probe_running_trainers,
    read_training_lock,
    run_serial_queue,
)

NAME = "20260930-p13-fold2-4090-candidate-resume-1h"
LOGS = ROOT / "artifacts/queues" / NAME
CONFIG = "configs/experiment/stage2_p13_fold2_sce_ac_14ep_4090_candidate.yaml"
OUTPUT = "artifacts/experiments/20260929-stage2-p13-fold2-sce-ac-14ep-4090-candidate"
RESUME = OUTPUT + "/last.pt"
RESUME_SHA = "67aa219049880da8f3eed694cc8a9752ed38eb5808a7cd635fde5e689b1d0cd6"
ORIGINAL_DISK_FREE = 11807141888
MAX_CUMULATIVE_ADDED = 4 * 1024**3
MIN_FREE = 5 * 1024**3
HARD_SECONDS = 3600
INPUTS = (*PAIR_INPUTS, "scripts/run_stage2_p13_fold2_4090_candidate_resume.py",
          "src/aictrack1/methods/confidence.py", RESUME)


def preflight() -> dict:
    if os.name != "posix" or sys.platform != "linux":
        raise RuntimeError("resume queue requires Linux")
    if LOGS.exists():
        raise FileExistsError(f"resume queue output already exists: {LOGS}")
    if read_training_lock(ROOT / DEFAULT_LOCK_RELATIVE_PATH) is not None or probe_running_trainers():
        raise RuntimeError("training lock or another supported trainer is active")
    if not torch.cuda.is_available() or "RTX 4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("resume requires the approved RTX 4090")
    gpu_free, gpu_total = torch.cuda.mem_get_info(0)
    disk_free, ram_free = shutil.disk_usage(ROOT).free, free_ram()
    if gpu_free < 20 * 1024**3 or disk_free < 9 * 1024**3 or ram_free < 10 * 1024**3:
        raise RuntimeError(f"resume resource guard failed: gpu={gpu_free}, disk={disk_free}, ram={ram_free}")
    if disk_free < max(MIN_FREE, ORIGINAL_DISK_FREE - MAX_CUMULATIVE_ADDED):
        raise RuntimeError("original cumulative disk budget already exhausted")
    output = ROOT / OUTPUT
    if not output.is_dir() or (output / "metrics.json").exists() or (output / "best.pt").exists():
        raise RuntimeError("candidate output is not the expected partial run")
    raw = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))["experiment"]
    if raw["training_scope"] != "fixed_split_validation" or raw["training"]["epochs"] != 14:
        raise RuntimeError("candidate training scope or total schedule drift")
    if raw["output_dir"] != "../../" + OUTPUT or raw["execution_enabled"] is not True:
        raise RuntimeError("candidate identity or execution approval drift")
    missing = [path for path in INPUTS if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"resume inputs missing: {missing}")
    frozen = {path: sha256(ROOT / path) for path in INPUTS}
    drift = {path: digest for path, digest in EXPECTED_SHA.items() if frozen[path] != digest}
    if drift or frozen[RESUME] != RESUME_SHA:
        raise RuntimeError(f"resume frozen input SHA mismatch: {drift}; resume={frozen[RESUME]}")
    state = torch.load(ROOT / RESUME, map_location="cpu", weights_only=True)
    if state.get("epoch") != 13 or len(state.get("history", ())) != 13:
        raise RuntimeError("resume checkpoint is not complete through epoch 13")
    expected_provenance = {
        "config_sha256": frozen[CONFIG],
        "train_manifest_sha256": EXPECTED_SHA["artifacts/audits/stage2/train/samples.csv"],
        "split_sha256": EXPECTED_SHA["artifacts/audits/stage2/train/split_fold2_seeded_random_20260924.csv"],
        "clip_weight_sha256": EXPECTED_SHA["artifacts/weights/ViT-B-32.pt"],
        "class_mapping_sha256": "811b767e5689fbed5ed9ed7f328e2e1458b02b208647f3ec1b3c3e99c7e4e879",
        "num_classes": 750,
        "visual_input_resolution": 336,
    }
    if any(state.get(key) != value for key, value in expected_provenance.items()):
        raise RuntimeError("resume checkpoint provenance mismatch")
    required = ("trainable_state", "best_state", "optimizer", "scheduler", "scaler",
                "python_rng_state", "numpy_rng_state", "torch_rng_state", "cuda_rng_state_all",
                "confidence_weighting")
    if any(key not in state for key in required):
        raise RuntimeError("resume checkpoint is missing optimizer/RNG/AC state")
    return {"scope": "Stage2 fold2 train-only candidate epoch13-to14 explicit resume",
            "gpu_name": torch.cuda.get_device_name(0), "gpu_free": gpu_free, "gpu_total": gpu_total,
            "disk_free": disk_free, "ram_free": ram_free,
            "original_disk_free": ORIGINAL_DISK_FREE,
            "max_cumulative_added_bytes": MAX_CUMULATIVE_ADDED,
            "budget_training_seconds": HARD_SECONDS, "resume_sha256": RESUME_SHA,
            "input_sha256": frozen}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="launch only this approved one-hour continuation")
    args = parser.parse_args()
    receipt = preflight()
    if not args.run:
        print(json.dumps({key: value for key, value in receipt.items() if key != "input_sha256"}, indent=2))
        return 0

    frozen = receipt["input_sha256"]
    started = datetime.now(ZoneInfo("Asia/Shanghai"))
    deadline = started + timedelta(seconds=HARD_SECONDS)
    LOGS.mkdir(parents=True, exist_ok=False)
    receipt.update({"started_at": started.isoformat(), "hard_deadline": deadline.isoformat(),
                    "queue_pid": os.getpid()})
    (LOGS / "source_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    def verify_frozen(*, include_resume: bool) -> None:
        changed = [path for path, digest in frozen.items()
                   if (include_resume or path != RESUME) and sha256(ROOT / path) != digest]
        if changed:
            raise RuntimeError(f"resume frozen inputs changed: {changed}")

    def launcher(command, cwd, output, on_start) -> int:
        verify_frozen(include_resume=True)
        if time.time() >= deadline.timestamp():
            return 124
        process = subprocess.Popen(list(command), cwd=cwd, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
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
                        or ORIGINAL_DISK_FREE - free > MAX_CUMULATIVE_ADDED):
                    output.write(f"P13 fold2 resume cutoff: free={free}; retaining atomic last.pt.\n")
                    output.flush()
                    stop_owned_process(process, output)
                    return 124

    item = QueueItem("candidate_epoch14", CONFIG, OUTPUT,
                     "explicit resume of same 14-epoch schedule from verified epoch13 last.pt",
                     resume_checkpoint=RESUME)
    code = run_serial_queue(root=ROOT, python=Path(sys.executable), logs=LOGS,
                            queue_name=NAME, items=[item], launcher=launcher)
    verify_frozen(include_resume=False)
    (LOGS / "queue_result.json").write_text(json.dumps({"exit_code": code,
        "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}, indent=2) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
