"""One bounded local Stage2 fold2 20-epoch schedule screen; never reads test.

The 14-epoch historical result is exploratory context only: it was trained on
another machine. No paired gate or full-training authorization is inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aictrack1.engine.training_queue import (  # noqa: E402
    DEFAULT_LOCK_RELATIVE_PATH, QueueItem, probe_running_trainers,
    read_training_lock, run_serial_queue,
)

NAME = "20261001-p21-fold2-local-20ep-screen"
LOGS = ROOT / "artifacts/queues" / NAME
CONFIG = "configs/experiment/stage2_p21_fold2_sce_ac_20ep_local_screen.yaml"
OUTPUT = "artifacts/experiments/20261001-stage2-p21-fold2-sce-ac-20ep-local-screen"
HARD_SECONDS = 7 * 3600
MAX_ADDED_BYTES = 3 * 1024**3
MIN_FREE_BYTES = 5 * 1024**3
EXPECTED = {
    "artifacts/audits/stage2/train/samples.csv": "1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d",
    "artifacts/audits/stage2/train/split_fold2_seeded_random_20260924.csv": "1e5eb7e9adbc234ba933bf76c99896602b8bdce5557879f90224ae2bb4cf2a50",
    "artifacts/audits/stage2/train/label_conflicts.csv": "b1afe084e654aeac5d431501657734c168a3cfc1f928defacac219dcf84c430a",
    "artifacts/audits/stage2/metadata/classes.csv": "0f54c471945f356177c885ba2f9edc943d669a61f12c73503fd4632c91ef40be",
    "artifacts/experiments/20260925-stage2-fold2-c5-h1-336-seed20260921/best.pt": "83380620f06af442ddbe4e55f048d6519d03ce95e672ab3bc51469904b803500",
    "artifacts/weights/ViT-B-32.pt": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def preflight() -> dict:
    if os.name != "nt":
        raise RuntimeError("P21 local queue requires Windows")
    if LOGS.exists() or (ROOT / OUTPUT).exists():
        raise FileExistsError("P21 queue or output already exists; no automatic restart")
    if read_training_lock(ROOT / DEFAULT_LOCK_RELATIVE_PATH) is not None or probe_running_trainers():
        raise RuntimeError("training lock or another trainer is active")
    if not torch.cuda.is_available() or "4060" not in torch.cuda.get_device_name(0):
        raise RuntimeError("approved local RTX 4060 unavailable")
    gpu_free, _ = torch.cuda.mem_get_info(0)
    disk_free = shutil.disk_usage(ROOT).free
    if gpu_free < 5 * 1024**3 or disk_free < 15 * 1024**3:
        raise RuntimeError(f"resource guard: gpu_free={gpu_free}, disk_free={disk_free}")
    config = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))["experiment"]
    training = config["training"]
    if (config.get("execution_enabled") is not True
            or config["training_scope"] != "fixed_split_validation"
            or config["competition_stage"] != "stage2"
            or config["run_id"] != Path(OUTPUT).name
            or (ROOT / CONFIG).parent.joinpath(config["output_dir"]).resolve() != (ROOT / OUTPUT).resolve()
            or training["epochs"] != 20 or training["batch_size"] != 64
            or training["seed"] != 20260921 or training["workers"] != 8
            or training["prefetch_factor"] != 4 or training["persistent_workers"] is not True
            or config["base_loss"] != {"kind": "sce", "alpha": 0.1, "beta": 1.0, "log_zero": -4.0}
            or config["peft"] != {"mode": "last_n_blocks", "lora_blocks": 8}
            or config["device"] != "cuda"):
        raise RuntimeError("P21 approved single-variable schedule/config mismatch")
    inputs = {CONFIG, "configs/data/stage2_fold2_336.yaml",
              "configs/model/openai_clip_vit_b32.yaml", "requirements-lock.txt",
              "scripts/train_peft.py", "scripts/run_stage2_p21_local_20ep_screen.py", *EXPECTED}
    inputs.update(path.relative_to(ROOT).as_posix() for path in (ROOT / "src").rglob("*.py"))
    missing = [name for name in inputs if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"P21 missing inputs: {missing}")
    frozen = {name: digest(ROOT / name) for name in sorted(inputs)}
    drift = {name: frozen[name] for name, expected in EXPECTED.items() if frozen[name] != expected}
    if drift:
        raise RuntimeError(f"P21 Stage2 source SHA mismatch: {drift}")
    if not (ROOT / "DataSetStage2/train").is_dir():
        raise FileNotFoundError("official Stage2 train directory missing")
    return {"scope": "Stage2 train-derived fold2 only; no test", "gpu": torch.cuda.get_device_name(0),
            "gpu_free_bytes": gpu_free, "disk_free_bytes": disk_free,
            "hard_seconds": HARD_SECONDS, "max_added_bytes": MAX_ADDED_BYTES,
            "input_sha256": frozen}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="consume approved 7h GPU batch; default read-only preflight")
    args = parser.parse_args()
    receipt = preflight()
    if not args.run:
        print(json.dumps({k: v for k, v in receipt.items() if k != "input_sha256"}, indent=2))
        return 0
    started = datetime.now(timezone.utc)
    deadline = started + timedelta(seconds=HARD_SECONDS)
    LOGS.mkdir(parents=True, exist_ok=False)
    receipt.update({"started_at": started.isoformat(), "deadline": deadline.isoformat(),
                    "queue_pid": os.getpid(), "authorization": "User approved local 20-epoch screen if server occupied, 2026-10-01",
                    "evaluation_status": "not authorized to infer paired gain without matched 14-epoch control"})
    save_json(LOGS / "source_receipt.json", receipt)

    def verify_inputs() -> None:
        changed = [name for name, expected in receipt["input_sha256"].items()
                   if digest(ROOT / name) != expected]
        if changed:
            raise RuntimeError(f"P21 frozen input changed: {changed}")

    def launch(command, cwd, output, attach) -> int:
        verify_inputs()
        process = subprocess.Popen(list(command), cwd=cwd, stdout=output,
                                   stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        attach(process.pid)
        next_status = 0.0
        while process.poll() is None:
            now = datetime.now(timezone.utc)
            free = shutil.disk_usage(ROOT).free
            if now >= deadline or free < MIN_FREE_BYTES or receipt["disk_free_bytes"] - free > MAX_ADDED_BYTES:
                output.write("P21 hard time/disk stop; preserving last complete atomic checkpoint.\n")
                output.flush()
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError("owned trainer termination unconfirmed; retain lock") from exc
                return 124
            if time.monotonic() >= next_status:
                last = ROOT / OUTPUT / "last.pt"
                save_json(LOGS / "status.json", {
                    "state": "running", "at": now.isoformat(), "trainer_pid": process.pid,
                    "deadline": deadline.isoformat(), "disk_free_bytes": free,
                    "last_bytes": last.stat().st_size if last.exists() else None,
                    "last_modified": last.stat().st_mtime if last.exists() else None,
                })
                next_status = time.monotonic() + 60
            time.sleep(5)
        return int(process.returncode)

    item = QueueItem("candidate_20ep", CONFIG, OUTPUT, "local fold2 20-epoch exploratory schedule screen")
    code = run_serial_queue(root=ROOT, python=Path(sys.executable), logs=LOGS,
                            queue_name=NAME, items=[item], launcher=launch)
    if code == 0:
        verify_inputs()
        metrics = json.loads((ROOT / OUTPUT / "metrics.json").read_text(encoding="utf-8"))
        if len(metrics["history"]) != 20 or metrics["history"][-1]["epoch"] != 20:
            raise RuntimeError("P21 20-epoch history incomplete")
        if any(row["confidence_weighting"]["next_effective_quality"]["decision"] == "L2"
               for row in metrics["history"]):
            raise RuntimeError("P21 AC quality L2")
        best = ROOT / OUTPUT / "best.pt"
        last = ROOT / OUTPUT / "last.pt"
        if digest(best) != metrics["selected_checkpoint_sha256"]:
            raise RuntimeError("P21 selected best SHA mismatch")
        state = torch.load(last, map_location="cpu", weights_only=True)
        required = {"trainable_state", "optimizer", "scheduler", "scaler", "epoch", "history",
                    "best_state", "python_rng_state", "numpy_rng_state", "torch_rng_state",
                    "cuda_rng_state_all", "confidence_weighting"}
        if not required.issubset(state) or state["epoch"] != 20:
            raise RuntimeError("P21 complete resumable last state missing")
        save_json(LOGS / "checkpoint_receipt.json", {"best_sha256": digest(best),
                    "last_sha256": digest(last), "last_weights_only_verified": True})
    save_json(LOGS / "queue_result.json", {"exit_code": code,
              "finished_at": datetime.now(timezone.utc).isoformat(),
              "paired_gate_passed": False, "test_access": False})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
