"""User-approved local-only P5 first-fold screen; no full training or test.

Reuse the repository atomic trainer guard. Freeze inputs before either run;
the two existing TTA diagnostics acquire their own repository lease afterward.
An eight-hour wall-clock budget includes training and train-derived audits.
No automatic resume, unknown-lock cleanup, second fold, or remote connection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aictrack1.engine.training_queue import QueueItem, run_serial_queue

QUEUE = "20260927-p5-local-fold1-pair"
LOGS = ROOT / "artifacts/queues" / QUEUE
CONFIGS = tuple(ROOT / f"configs/experiment/stage2_p5_sce_fold1_local_{tag}.yaml"
                for tag in ("control", "candidate"))
LOSSES = ({"kind": "gce", "q": .7},
          {"kind": "sce", "alpha": .1, "beta": 1., "log_zero": -4.})
THRESHOLDS = {"top1_delta_min": .002, "ci95_low_exclusive_min": 0.,
              "macro_delta_exclusive_min": 0., "tail_delta_min": -.0025}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_pair(configs=CONFIGS) -> list[QueueItem]:
    experiments = [yaml.safe_load(path.read_text(encoding="utf-8"))["experiment"] for path in configs]
    control, candidate = experiments
    ignored = {"run_id", "output_dir", "base_loss"}
    if {k: v for k, v in control.items() if k not in ignored} != {k: v for k, v in candidate.items() if k not in ignored}:
        raise ValueError("paired recipes must differ only in loss and identity")
    items = []
    for path, ex, loss in zip(configs, experiments, LOSSES, strict=True):
        if ex.get("execution_enabled") is not True or ex.get("competition_stage") != "stage2":
            raise ValueError("local approved Stage2 configuration required")
        if ex.get("training_scope") != "fixed_split_validation" or ex.get("base_loss") != loss:
            raise ValueError("only the preregistered loss screen is approved")
        training = ex["training"]
        if (training["epochs"], training["batch_size"], training["seed"], training["workers"]) != (8, 64, 20260921, 4):
            raise ValueError("unapproved training budget or recipe")
        output = (path.parent / ex["output_dir"]).resolve()
        if output.parent != ROOT / "artifacts/experiments" or "-p5-fold1-local-" not in output.name:
            raise ValueError("requires unique local P5 output")
        items.append(QueueItem(tag=ex["run_id"], config=path.relative_to(ROOT).as_posix(),
                               output_dir=output.relative_to(ROOT).as_posix(), purpose="local P5 matched 8-epoch loss screen"))
    if items[0].output_dir == items[1].output_dir:
        raise ValueError("paired outputs must be distinct")
    return items


def input_hashes() -> dict[str, str]:
    paths = set(CONFIGS)
    for directory in (ROOT / "src", ROOT / "scripts"):
        paths.update(directory.rglob("*.py"))
    paths.update((ROOT / "requirements-lock.txt", ROOT / "configs/data/stage2_fold1_336.yaml",
                  ROOT / "configs/model/openai_clip_vit_b32.yaml", ROOT / "artifacts/weights/ViT-B-32.pt",
                  ROOT / "artifacts/audits/stage2/metadata/classes.csv"))
    ex = yaml.safe_load(CONFIGS[0].read_text(encoding="utf-8"))["experiment"]
    for field in ("manifest", "split", "label_conflicts", "head_checkpoint"):
        paths.add((CONFIGS[0].parent / ex[field]).resolve())
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths)}


def require_space(min_gib: int = 3) -> None:
    if shutil.disk_usage(ROOT).free < min_gib * 1024**3:
        raise RuntimeError(f"requires {min_gib} GiB free; no automatic deletion")


def verify_owned_last(path: Path, config_digest: str) -> dict:
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {"trainable_state", "optimizer", "scheduler", "scaler", "epoch", "history", "best_state",
                "python_rng_state", "numpy_rng_state", "torch_rng_state", "cuda_rng_state_all", "confidence_weighting"}
    if not required.issubset(state) or state["config_sha256"] != config_digest or state["competition_stage"] != "stage2":
        raise RuntimeError("local last lacks complete approved state")
    if not state["optimizer"]["state"] or not state["trainable_state"] or not state["scheduler"] or not state["scaler"]:
        raise RuntimeError("local last optimizer/scheduler/scaler/model state empty")
    return {"epoch": state["epoch"], "sha256": sha256(path), "bytes": path.stat().st_size,
            "cpu_weights_only_load_verified": True, "complete_state_keys_present": True,
            "end_to_end_resume_equivalence_verified": False}


def pair_checks(reports: list[dict], metrics: list[dict], audit: dict) -> dict:
    # Ratios, not percentage-point values. Diagnostic's older internal TTA gate
    # is not a loss-comparison gate and is deliberately not used here.
    base, cand = (r["four_view"] for r in reports)
    delta = {key: cand[key] - base[key] for key in ("top1", "macro_accuracy", "tail_class_mean_recall")}
    checks = {
        "top1": delta["top1"] >= THRESHOLDS["top1_delta_min"],
        "ci95_low": audit["paired_near_duplicate_group_bootstrap"]["ci95_low"] > 0,
        "macro": delta["macro_accuracy"] > 0,
        "tail": delta["tail_class_mean_recall"] >= THRESHOLDS["tail_delta_min"],
        "complete_count": all(r["evaluated_count"] == 29737 and not r["smoke_only"] and r["four_view_resize"] == 384 for r in reports),
        "eight_epochs": all(len(m["history"]) == 8 and m["history"][-1]["epoch"] == 8 for m in metrics),
        "ac_non_l2": all(row["confidence_weighting"]["next_effective_quality"]["decision"] != "L2"
                         for m in metrics for row in m["history"]),
        "center_matches": all(abs(r["center"]["top1"] - m["best_top1"]) <= 1e-6 for r, m in zip(reports, metrics, strict=True)),
        "gradient_diagnostics_present": all("gradient_diagnostics" in row for m in metrics for row in m["history"]),
    }
    return {"thresholds": THRESHOLDS, "delta": delta, "checks": checks, "passed": all(checks.values()),
            "full_training_authorized": False, "next_fold_authorized": False,
            "caveat": "selected noisy validation fold; overlapping folds; not hidden-test or seed uncertainty"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("this approved queue is local Windows only")
    items = validate_pair()
    require_space(8)
    frozen = input_hashes()
    if frozen["artifacts/weights/ViT-B-32.pt"] != "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af":
        raise RuntimeError("official CLIP weight hash mismatch")
    if frozen["artifacts/audits/stage2/train/samples.csv"] != "1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d" or frozen["artifacts/audits/stage2/train/split.csv"] != "c8cd505ad24400546d4e2ac78ba26d313e08b96d4f41e0235960e33da343b03c":
        raise RuntimeError("Stage2 fixed fold inputs changed")
    import torch
    if not torch.cuda.is_available() or "4060" not in torch.cuda.get_device_name(0):
        raise RuntimeError("approved local RTX4060 unavailable")
    print(json.dumps({"check": "passed", "device": torch.cuda.get_device_name(0), "input_file_count": len(frozen), "free_gib": shutil.disk_usage(ROOT).free / 1024**3}), flush=True)
    if args.check_only:
        return 0
    LOGS.mkdir(parents=True, exist_ok=True)
    receipt_path = LOGS / "approval_receipt.json"
    start = datetime.now(timezone.utc)
    deadline = start + timedelta(hours=8)
    receipt = {"schema": "aictrack1.p5.local_screen.v1", "authorization": "User explicitly approved continuation on local machine only, 2026-09-27",
               "started_at": start.isoformat(), "deadline": deadline.isoformat(), "wall_clock_cap_hours": 8,
               "source_and_input_sha256": frozen, "thresholds": THRESHOLDS, "remote_access": False,
               "train_scope": "Stage2 train-derived fold1 only", "test_access": False,
               "next_fold_authorized": False, "full_training_authorized": False}
    # A new invocation cannot silently reset the elapsed budget or rerun a partial batch.
    with receipt_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    next_snapshot = time.monotonic() + 3600
    snapshot_count = 0

    def launch(command, cwd, log, attach):
        nonlocal next_snapshot, snapshot_count
        require_space()
        if datetime.now(timezone.utc) >= deadline or input_hashes() != frozen:
            raise RuntimeError("budget exhausted or frozen source/inputs changed")
        proc = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        attach(proc.pid)
        next_status = 0.
        while proc.poll() is None:
            now = datetime.now(timezone.utc)
            free = shutil.disk_usage(ROOT).free
            if now >= deadline or free < 3 * 1024**3:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    # Unknown live child => the caller retains its repository lease.
                    raise RuntimeError("child termination unconfirmed; inspect before recovery")
                write_json(LOGS / "status.json", {"state": "budget_or_disk_stop", "child_pid": proc.pid, "at": now.isoformat(), "returncode": proc.returncode})
                return 124
            if time.monotonic() >= next_status:
                checkpoints = {}
                for item in items:
                    path = ROOT / item.output_dir / "last.pt"
                    if path.exists():
                        stat = path.stat()
                        checkpoints[item.tag] = {"bytes": stat.st_size, "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}
                write_json(LOGS / "status.json", {"state": "running", "at": now.isoformat(), "child_pid": proc.pid,
                           "command": list(command), "deadline": deadline.isoformat(), "free_gib": free / 1024**3,
                           "canonical_last": checkpoints, "checkpoint_cpu_restore_verified": False})
                next_status = time.monotonic() + 60
            if time.monotonic() >= next_snapshot and snapshot_count < 7:
                snapshot_count += 1
                for item in items:
                    path = ROOT / item.output_dir / "last.pt"
                    if not path.exists(): continue
                    require_space(4)
                    target = LOGS / "snapshots" / f"hour{snapshot_count}-{item.tag}-last.pt"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    # Atomic trainer replacement makes a hard link a stable inode
                    # snapshot; never copy an actively replaced pathname in chunks.
                    os.link(path, target)
                    audit = verify_owned_last(target, frozen[item.config])
                    audit.update({"source": str(path), "restore_path": str(target), "verified_at": datetime.now(timezone.utc).isoformat()})
                    write_json(target.with_suffix(".json"), audit)
                next_snapshot = time.monotonic() + 3600
            time.sleep(5)
        if proc.returncode == 0 and "scripts/train_peft.py" in command:
            config_path = ROOT / command[command.index("--config") + 1]
            ex = yaml.safe_load(config_path.read_text(encoding="utf-8"))["experiment"]
            output = (config_path.parent / ex["output_dir"]).resolve()
            metric = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            if len(metric["history"]) != 8 or metric["history"][-1]["epoch"] != 8 or metric.get("early_stop"):
                return 3
            if any(row["confidence_weighting"]["next_effective_quality"]["decision"] == "L2" for row in metric["history"]):
                return 3
            audit = verify_owned_last(output / "last.pt", frozen[config_path.relative_to(ROOT).as_posix()])
            write_json(LOGS / f"{ex['run_id']}-final-last-verification.json", audit)
        return proc.returncode

    code = run_serial_queue(root=ROOT, python=Path(sys.executable), logs=LOGS,
                            queue_name=QUEUE, items=items, launcher=launch)
    if code:
        write_json(LOGS / "status.json", {"state": "training_halted", "returncode": code, "at": datetime.now(timezone.utc).isoformat()})
        return code
    metrics, reports, diagnostic_paths = [], [], []
    for tag, item in zip(("control", "candidate"), items, strict=True):
        output = ROOT / item.output_dir
        metric = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        metrics.append(metric)
        best = output / "best.pt"
        digest = sha256(best)
        if digest != metric["selected_checkpoint_sha256"] or metric["selected_checkpoint"] != "best.pt":
            raise RuntimeError("selected best checkpoint hash mismatch")
        diagnostic = ROOT / "artifacts/diagnostics" / f"{QUEUE}-{tag}-384-fourview"
        diagnostic_paths.append(diagnostic)
        command = (sys.executable, "-u", "scripts/diagnose_stage2_tta.py", "--experiment-config", item.config,
                   "--checkpoint", str(best), "--expected-checkpoint-sha256", digest,
                   "--output-dir", str(diagnostic), "--batch-size", "64", "--workers", "4", "--four-view-scale384")
        with (LOGS / f"{tag}-tta.log").open("x", encoding="utf-8") as handle:
            code = launch(command, ROOT, handle, lambda pid: None)  # diagnostic owns its own atomic lease
        if code:
            write_json(LOGS / "status.json", {"state": "diagnostic_halted", "returncode": code, "tag": tag})
            return code
        reports.append(json.loads((diagnostic / "report.json").read_text(encoding="utf-8")))
    audit_path = ROOT / "artifacts/reports" / f"{QUEUE}-384-fourview-bootstrap.json"
    command = (sys.executable, "-u", "scripts/audit_stage2_paired_tta_gain.py",
               "--baseline-predictions", str(diagnostic_paths[0] / "paired_predictions.csv"),
               "--candidate-predictions", str(diagnostic_paths[1] / "paired_predictions.csv"),
               "--baseline-column", "four_view_prediction", "--candidate-column", "four_view_prediction",
               "--replicates", "500", "--seed", "20260925", "--output", str(audit_path))
    with (LOGS / "bootstrap.log").open("x", encoding="utf-8") as handle:
        code = launch(command, ROOT, handle, lambda pid: None)
    if code:
        return code
    result = pair_checks(reports, metrics, json.loads(audit_path.read_text(encoding="utf-8")))
    result.update({"approval_receipt_sha256": sha256(receipt_path), "completed_at": datetime.now(timezone.utc).isoformat(),
                   "audit_sha256": sha256(audit_path), "reports_sha256": [sha256(p / "report.json") for p in diagnostic_paths]})
    write_json(LOGS / "screen_result.json", result)
    write_json(LOGS / "status.json", {"state": "screen_complete", "passed": result["passed"], "at": result["completed_at"]})
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
