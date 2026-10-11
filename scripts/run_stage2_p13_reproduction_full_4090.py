"""Bounded one-arm P13/P5 full-14 reproducibility queue on the approved RTX 4090.

No test access. Default is a read-only source/resource preflight; --run starts
exactly one atomic-lock-protected training item in a unique directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aictrack1.engine.training_queue import (  # noqa: E402
    DEFAULT_LOCK_RELATIVE_PATH,
    QueueItem,
    probe_running_trainers,
    read_training_lock,
    run_serial_queue,
)

NAME = "20260930-p13-reproduction-full14-4090"
CONFIG = "configs/experiment/stage2_p13_reproduction_sce_ac_full14_4090.yaml"
OUTPUT = "artifacts/experiments/20260930-stage2-p13-reproduction-sce-ac-full14-4090"
LOGS = ROOT / "artifacts/queues" / NAME
HARD_SECONDS = 5 * 3600
MAX_ADDED_BYTES = 4 * 1024**3
MIN_FREE_BYTES = 5 * 1024**3
INPUTS = (
    "scripts/train_peft.py",
    "scripts/run_stage2_p13_reproduction_full_4090.py",
    "src/aictrack1/engine/training_queue.py",
    "src/aictrack1/methods/sce.py",
    "configs/data/stage2_fold1_336.yaml",
    "configs/model/openai_clip_vit_b32.yaml",
    "configs/experiment/stage2_p5_sce_peft_336_full_14ep_a10.yaml",
    CONFIG,
    "artifacts/audits/stage2/train/samples.csv",
    "artifacts/audits/stage2/train/split.csv",
    "artifacts/audits/stage2/train/label_conflicts.csv",
    "artifacts/audits/stage2/metadata/classes.csv",
    "artifacts/experiments/20260928-stage2-p5-sce-h1-336-full-seed20260921/final.pt",
    "artifacts/weights/ViT-B-32.pt",
)
EXPECTED = {
    "configs/data/stage2_fold1_336.yaml": "260442564226ce124dee444b7a81460d65314b3bdc930a2859b503b065653807",
    "artifacts/audits/stage2/train/samples.csv": "1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d",
    "artifacts/audits/stage2/train/split.csv": "c8cd505ad24400546d4e2ac78ba26d313e08b96d4f41e0235960e33da343b03c",
    "artifacts/audits/stage2/train/label_conflicts.csv": "b1afe084e654aeac5d431501657734c168a3cfc1f928defacac219dcf84c430a",
    "artifacts/audits/stage2/metadata/classes.csv": "0f54c471945f356177c885ba2f9edc943d669a61f12c73503fd4632c91ef40be",
    "artifacts/experiments/20260928-stage2-p5-sce-h1-336-full-seed20260921/final.pt": "b17d08319da4a622b29a603cc2b88382cd1b79500b5aac377e4b0b47d54c2a3e",
    "artifacts/weights/ViT-B-32.pt": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def free_ram() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("cannot read MemAvailable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("4090 queue is Linux-only")
    if LOGS.exists() or (ROOT / OUTPUT).exists():
        raise FileExistsError("reproduction queue/output already exists")
    if read_training_lock(ROOT / DEFAULT_LOCK_RELATIVE_PATH) is not None or probe_running_trainers():
        raise RuntimeError("training lock or trainer active")
    if not torch.cuda.is_available() or "RTX 4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("approved RTX 4090 unavailable")
    gpu_free, gpu_total = torch.cuda.mem_get_info(0)
    disk_free, ram_free = shutil.disk_usage(ROOT).free, free_ram()
    if gpu_free < 20 * 1024**3 or disk_free < 8 * 1024**3 or ram_free < 10 * 1024**3:
        raise RuntimeError(f"resource guard: gpu={gpu_free}, disk={disk_free}, ram={ram_free}")
    if any(not (ROOT / relative).is_file() for relative in INPUTS):
        raise FileNotFoundError("a frozen full-train input is missing")
    frozen = {relative: sha256(ROOT / relative) for relative in INPUTS}
    if any(frozen[relative] != expected for relative, expected in EXPECTED.items()):
        raise RuntimeError("frozen source SHA mismatch")
    current = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))["experiment"]
    old = yaml.safe_load((ROOT / "configs/experiment/stage2_p5_sce_peft_336_full_14ep_a10.yaml").read_text(encoding="utf-8"))["experiment"]
    if {k: v for k, v in current.items() if k not in {"run_id", "output_dir"}} != {k: v for k, v in old.items() if k not in {"run_id", "output_dir"}}:
        raise RuntimeError("full-train recipe differs from P5 full-14 baseline")
    if (current["training_scope"] != "full_train_fixed_epochs" or current["training"]["epochs"] != 14 or
            current["expected_train_count"] != 148695 or current["device"] != "cuda"):
        raise RuntimeError("scope, epochs, count or device drift")
    train_root = ROOT / "DataSetStage2/train"
    if not train_root.is_dir() or sum(1 for _ in train_root.rglob("*.jpg")) != 148695:
        raise RuntimeError("Stage2 train root/count mismatch")
    state = {"scope": "Stage2 full-train reproduction, no test", "run": args.run,
             "gpu_free": gpu_free, "gpu_total": gpu_total, "disk_free": disk_free,
             "ram_free": ram_free, "input_sha256": frozen,
             "budget_seconds": HARD_SECONDS, "budget_added_bytes": MAX_ADDED_BYTES}
    if not args.run:
        print(json.dumps({k: v for k, v in state.items() if k != "input_sha256"}, indent=2))
        return 0

    started = datetime.now(ZoneInfo("Asia/Shanghai"))
    deadline = started + timedelta(seconds=HARD_SECONDS)
    LOGS.mkdir(parents=True, exist_ok=False)
    state.update({"started_at": started.isoformat(), "hard_deadline": deadline.isoformat(), "queue_pid": os.getpid()})
    (LOGS / "source_receipt.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    def launcher(command, cwd, output, on_start):
        if any(sha256(ROOT / relative) != digest for relative, digest in frozen.items()):
            raise RuntimeError("frozen source changed before launch")
        if time.time() >= deadline.timestamp():
            return 124
        child = subprocess.Popen(list(command), cwd=cwd, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        on_start(child.pid)
        while True:
            try:
                return child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                free = shutil.disk_usage(ROOT).free
                if time.time() >= deadline.timestamp() or free < MIN_FREE_BYTES or disk_free - free > MAX_ADDED_BYTES:
                    output.write(f"P13 full hard cutoff: free={free}, added={disk_free-free}; keep atomic last.pt.\n")
                    output.flush()
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=30)
                    return 124

    code = run_serial_queue(root=ROOT, python=Path(sys.executable), logs=LOGS, queue_name=NAME,
                            items=[QueueItem("full14", CONFIG, OUTPUT, "P13/P5 single-model full-14 reproduction")],
                            launcher=launcher)
    (LOGS / "queue_result.json").write_text(json.dumps({"exit_code": code,
        "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}, indent=2) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
