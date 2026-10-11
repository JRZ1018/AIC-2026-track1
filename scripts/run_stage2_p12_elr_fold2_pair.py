"""Bounded local RTX4060 SCE-vs-ELR pair with train-only diagnostics."""

from __future__ import annotations

import csv
import ctypes
import hashlib
import json
import shutil
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
    TrainingLease,
    probe_running_trainers,
    read_training_lock,
    run_serial_queue,
)

QUEUE_NAME = "20260929-p12-local-fold2-elr-pair"
QUEUE_DIR = ROOT / "artifacts/queues" / QUEUE_NAME
CONTROL = "configs/experiment/stage2_p12_fold2_sce_control.yaml"
CANDIDATE = "configs/experiment/stage2_p12_fold2_elr_candidate.yaml"
CONTROL_RUN = "artifacts/experiments/20260929-stage2-p12-fold2-sce-control"
CANDIDATE_RUN = "artifacts/experiments/20260929-stage2-p12-fold2-elr-candidate"
CONTROL_DIAG = "artifacts/diagnostics/20260929-p12-fold2-sce-control-384-fourview"
CANDIDATE_DIAG = "artifacts/diagnostics/20260929-p12-fold2-elr-candidate-384-fourview"
BOOTSTRAP = "artifacts/reports/20260929-p12-fold2-elr-vs-sce-paired-bootstrap.json"
SAMPLES = "artifacts/audits/stage2/train/samples.csv"
CONFLICTS = "artifacts/audits/stage2/train/label_conflicts.csv"
SOURCE_INPUTS = (
    "scripts/run_stage2_p12_elr_fold2_pair.py",
    "scripts/train_peft.py",
    "scripts/diagnose_stage2_tta.py",
    "scripts/audit_stage2_paired_tta_gain.py",
    "src/aictrack1/engine/training_queue.py",
    "src/aictrack1/methods/elr.py",
    "src/aictrack1/methods/sce.py",
    "tests/test_elr.py",
    "tests/test_train_peft_elr.py",
    "docs/04_上游方案调研与复用登记.md",
    "docs/05_上游最小复用规格.md",
    "docs/157_P12_ELR复赛单折配对短筛预注册_20260929.md",
    "docs/upstream_registry.yaml",
    "configs/data/stage2_fold2_336.yaml",
    "configs/model/openai_clip_vit_b32.yaml",
    "configs/experiment/stage2_p12_fold2_sce_control.yaml",
    "configs/experiment/stage2_p12_fold2_elr_candidate.yaml",
    SAMPLES,
    "artifacts/audits/stage2/train/split_fold2_seeded_random_20260924.csv",
    CONFLICTS,
    "artifacts/experiments/20260925-stage2-fold2-c5-h1-336-seed20260921/best.pt",
    "artifacts/weights/ViT-B-32.pt",
)
OUTPUTS = (CONTROL_RUN, CANDIDATE_RUN)
DEADLINE_EPOCHS = 8
HARD_BUDGET_HOURS = 6.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _ram_available_bytes() -> int:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return int(status.ullAvailPhys)


def _validate_configs() -> None:
    first = yaml.safe_load((ROOT / CONTROL).read_text(encoding="utf-8"))["experiment"]
    second = yaml.safe_load((ROOT / CANDIDATE).read_text(encoding="utf-8"))["experiment"]
    changed = {key for key in first if first[key] != second[key]}
    if changed != {"run_id", "output_dir", "base_loss"}:
        raise RuntimeError(f"P12 pair has unintended config differences: {sorted(changed)}")
    if first["base_loss"] != {"kind": "sce", "alpha": 0.1, "beta": 1.0, "log_zero": -4.0}:
        raise RuntimeError("P12 control must remain the frozen SCE baseline")
    if second["base_loss"] != {"kind": "elr", "beta": 0.7, "lambda": 3.0, "eps": 1e-6}:
        raise RuntimeError("P12 ELR candidate differs from the preregistered fixed values")
    for config in (first, second):
        if config.get("execution_enabled") is not True:
            raise RuntimeError("P12 configs must be explicitly enabled")
        if config.get("training_scope") != "fixed_split_validation":
            raise RuntimeError("P12 is restricted to the fixed Stage2 train split")
        if config["training"]["epochs"] != DEADLINE_EPOCHS or config["training"]["seed"] != 20260921:
            raise RuntimeError("P12 schedule or seed drifted")
        if config["visual_input_resolution"] != 336 or config["peft"] != {"mode": "last_n_blocks", "lora_blocks": 8}:
            raise RuntimeError("P12 model capacity or resolution drifted")
        expected = {
            "dataset_config": "../data/stage2_fold2_336.yaml",
            "model_config": "../model/openai_clip_vit_b32.yaml",
            "manifest": "../../artifacts/audits/stage2/train/samples.csv",
            "split": "../../artifacts/audits/stage2/train/split_fold2_seeded_random_20260924.csv",
            "label_conflicts": "../../artifacts/audits/stage2/train/label_conflicts.csv",
            "head_checkpoint": "../../artifacts/experiments/20260925-stage2-fold2-c5-h1-336-seed20260921/best.pt",
            "device": "cuda",
            "position_interpolation": "bicubic_7x7_to_10x10",
            "expected_initial_top1": 0.6615327596664429,
            "initial_top1_tolerance": 0.002,
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise RuntimeError(f"P12 {key} drifted from the registered fold2 setup")
        expected_runtime = {
            "defer_epoch_metrics": True,
            "reuse_validation_loader": True,
            "record_gradient_diagnostics": True,
            "safe_numpy_rng_checkpoint": True,
        }
        if config.get("runtime") != expected_runtime:
            raise RuntimeError("P12 runtime settings drifted from the matched trainer setup")
        expected_confidence = {
            "enabled": True,
            "warmup_epochs": 1,
            "momentum": 0.8,
            "history_mix": 0.5,
            "min_weight": 0.02,
            "magnitude_mix": 0.75,
            "stop_on_effective_quality_l2": True,
        }
        if config.get("confidence_weighting") != expected_confidence:
            raise RuntimeError("P12 AC settings drifted from the matched fold2 setup")
        expected_training = {
            "epochs": 8,
            "batch_size": 64,
            "workers": 4,
            "prefetch_factor": 4,
            "persistent_workers": True,
            "seed": 20260921,
            "visual_learning_rate": 0.00000175,
            "head_learning_rate": 0.0,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
        }
        if config.get("training") != expected_training:
            raise RuntimeError("P12 optimizer/loader settings drifted from the matched fold2 setup")
    if first["run_id"] != "20260929-stage2-p12-fold2-sce-control" or second["run_id"] != "20260929-stage2-p12-fold2-elr-candidate":
        raise RuntimeError("P12 run IDs drifted")


def _finite_tensors(value: object) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_finite_tensors(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tensors(item) for item in value)
    return True


def _validate_run(run_rel: str, *, expect_elr: bool, expected_run_id: str) -> dict[str, object]:
    run = ROOT / run_rel
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    best_path, last_path = run / "best.pt", run / "last.pt"
    best_sha, last_sha = _sha256(best_path), _sha256(last_path)
    if metrics.get("run_id") != expected_run_id or metrics.get("selected_checkpoint") != "best.pt":
        raise RuntimeError(f"unexpected run identity/selection in {run_rel}")
    if metrics.get("selected_checkpoint_sha256") != best_sha or metrics.get("best_checkpoint_sha256") != best_sha:
        raise RuntimeError(f"best checkpoint SHA mismatch in {run_rel}")
    if metrics.get("last_checkpoint_sha256") != last_sha:
        raise RuntimeError(f"last checkpoint SHA mismatch in {run_rel}")
    if metrics.get("train_count") != 118958 or metrics.get("validation_count") != 29737:
        raise RuntimeError(f"unexpected Stage2 split counts in {run_rel}")
    if len(metrics.get("history", [])) != DEADLINE_EPOCHS:
        raise RuntimeError(f"incomplete epoch history in {run_rel}")
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    last = torch.load(last_path, map_location="cpu", weights_only=True)
    for name in ("trainable_state",):
        if name not in best or name not in last:
            raise RuntimeError(f"{run_rel} checkpoint missing {name}")
    required_last = {
        "optimizer", "scheduler", "scaler", "python_rng_state", "numpy_rng_state",
        "torch_rng_state", "cuda_rng_state_all", "confidence_weighting", "epoch", "history",
    }
    if not required_last.issubset(last):
        raise RuntimeError(f"{run_rel} last checkpoint is missing resume fields")
    if int(last["epoch"]) != DEADLINE_EPOCHS:
        raise RuntimeError(f"{run_rel} last checkpoint is not the final epoch")
    if expect_elr:
        state = last.get("elr_state")
        if not isinstance(state, dict) or not torch.is_tensor(state.get("target")):
            raise RuntimeError("ELR last checkpoint has no safe target state tensor")
        if tuple(state["target"].shape) != (148695, 750):
            raise RuntimeError("ELR state shape does not match the Stage2 train manifest")
    elif "elr_state" in last:
        raise RuntimeError("SCE control unexpectedly contains ELR state")
    if not _finite_tensors(best) or not _finite_tensors(last):
        raise FloatingPointError(f"non-finite checkpoint tensor found in {run_rel}")
    decisions = []
    for row in metrics["history"]:
        item = row.get("confidence_weighting", {}).get("next_effective_quality", {})
        decision = item.get("decision") or row.get("effective_quality_decision")
        decisions.append(decision)
    return {
        "run_id": expected_run_id,
        "best_sha256": best_sha,
        "last_sha256": last_sha,
        "best_weights_only_cpu_load": True,
        "last_weights_only_cpu_load": True,
        "last_epoch": int(last["epoch"]),
        "best_top1": float(metrics["best_top1"]),
        "config_sha256": str(metrics["config_sha256"]),
        "manifest_sha256": str(metrics["manifest_sha256"]),
        "split_sha256": str(metrics["split_sha256"]),
        "required_resume_fields": sorted(required_last),
        "elr_state_shape": [148695, 750] if expect_elr else None,
        "ac_decisions": decisions,
        "train_count": metrics["train_count"],
        "validation_count": metrics["validation_count"],
    }


def _taskkill_and_confirm(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    subprocess.run(
        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"could not confirm exit of owned child PID {process.pid}; retain lock") from exc


def _deadline_launcher(deadline_epoch: float):
    def launch(command, cwd, output, on_start) -> int:
        remaining = deadline_epoch - time.time()
        if remaining <= 0:
            output.write("P12 hard deadline reached before child launch; no child started.\n")
            output.flush()
            return 124
        process = subprocess.Popen(list(command), cwd=cwd, stdout=output, stderr=subprocess.STDOUT)
        try:
            on_start(process.pid)
        except Exception:
            _taskkill_and_confirm(process)
            raise
        try:
            return process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            output.write("P12 hard deadline: stopping only this owned trainer process tree; latest atomic last.pt is retained.\n")
            output.flush()
            _taskkill_and_confirm(process)
            return 124

    return launch


def _run_locked_diagnostics(deadline_epoch: float, checks: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    control_sha = str(checks["control"]["best_sha256"])
    candidate_sha = str(checks["candidate"]["best_sha256"])
    commands = [
        (
            "diagnose_control",
            [sys.executable, "scripts/diagnose_stage2_tta.py", "--experiment-config", CONTROL,
             "--checkpoint", f"{CONTROL_RUN}/best.pt", "--expected-checkpoint-sha256", control_sha,
             "--output-dir", CONTROL_DIAG, "--batch-size", "64", "--workers", "4",
             "--four-view-scale384", "--save-fused-logits"],
            QUEUE_DIR / "diagnose_control.log",
        ),
        (
            "diagnose_candidate",
            [sys.executable, "scripts/diagnose_stage2_tta.py", "--experiment-config", CANDIDATE,
             "--checkpoint", f"{CANDIDATE_RUN}/best.pt", "--expected-checkpoint-sha256", candidate_sha,
             "--output-dir", CANDIDATE_DIAG, "--batch-size", "64", "--workers", "4",
             "--four-view-scale384", "--save-fused-logits"],
            QUEUE_DIR / "diagnose_candidate.log",
        ),
        (
            "paired_bootstrap",
            [sys.executable, "scripts/audit_stage2_paired_tta_gain.py",
             "--baseline-predictions", f"{CONTROL_DIAG}/paired_predictions.csv",
             "--candidate-predictions", f"{CANDIDATE_DIAG}/paired_predictions.csv",
             "--baseline-column", "four_view_prediction", "--candidate-column", "four_view_prediction",
             "--output", BOOTSTRAP, "--seed", "20260925", "--replicates", "500"],
            QUEUE_DIR / "paired_bootstrap.log",
        ),
    ]
    if any((ROOT / rel).exists() for rel in (CONTROL_DIAG, CANDIDATE_DIAG, BOOTSTRAP)):
        raise FileExistsError("P12 train-only diagnostic output already exists; refusing overwrite")
    lock_path = ROOT / DEFAULT_LOCK_RELATIVE_PATH
    for tag, command, log_path in commands[:2]:
        if time.time() >= deadline_epoch:
            raise TimeoutError("P12 hard deadline reached before train-only diagnostics completed")
        if read_training_lock(lock_path) is not None or probe_running_trainers():
            raise RuntimeError(f"P12 {tag} preflight found an occupied or ambiguous training lock")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            remaining = deadline_epoch - time.time()
            if remaining <= 0:
                _taskkill_and_confirm(process)
                raise TimeoutError("P12 hard deadline reached after diagnostic child launch")
            try:
                return_code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                _taskkill_and_confirm(process)
                # The diagnostic owns its own lease. After force termination,
                # leave any surviving lease untouched for manual inspection.
                raise TimeoutError(f"P12 hard deadline stopped owned child {tag}")
        if return_code != 0:
            raise RuntimeError(f"P12 diagnostic {tag} failed with exit code {return_code}")
        if read_training_lock(lock_path) is not None or probe_running_trainers():
            raise RuntimeError(f"P12 {tag} exited but its shared lease/process state is not clear")

    if time.time() >= deadline_epoch:
        raise TimeoutError("P12 hard deadline reached before train-only paired bootstrap")
    tag, command, log_path = commands[2]
    lease = TrainingLease.acquire(
        lock_path,
        owner=f"{QUEUE_NAME}-{tag}",
        config="P12 CPU train-only paired near-duplicate bootstrap",
        output_dir=BOOTSTRAP,
        command=command,
    )
    process: subprocess.Popen | None = None
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            remaining = deadline_epoch - time.time()
            if remaining <= 0:
                _taskkill_and_confirm(process)
                raise TimeoutError("P12 hard deadline reached after bootstrap launch")
            try:
                return_code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                _taskkill_and_confirm(process)
                raise TimeoutError("P12 hard deadline stopped owned bootstrap child")
        if return_code != 0:
            raise RuntimeError(f"P12 bootstrap failed with exit code {return_code}")
    finally:
        # Release only this exact lease, and only after the owned child has exited.
        if process is not None and process.poll() is not None:
            lease.release()
    reports = {
        "control": json.loads((ROOT / CONTROL_DIAG / "report.json").read_text(encoding="utf-8")),
        "candidate": json.loads((ROOT / CANDIDATE_DIAG / "report.json").read_text(encoding="utf-8")),
    }
    audit = json.loads((ROOT / BOOTSTRAP).read_text(encoding="utf-8"))
    return reports, audit


def _ordinary_delta() -> dict[str, object]:
    metadata = {}
    with (ROOT / SAMPLES).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["source_split"] != "train":
                raise ValueError("P12 manifest includes a non-train source")
            metadata[row["sample_id"]] = (row["sha256"], row["internal_class_index"])
    with (ROOT / CONFLICTS).open("r", encoding="utf-8", newline="") as handle:
        conflict_hashes = {row["sha256"] for row in csv.DictReader(handle)}
    with (ROOT / f"{CONTROL_DIAG}/paired_predictions.csv").open("r", encoding="utf-8", newline="") as handle:
        control_rows = list(csv.DictReader(handle))
    with (ROOT / f"{CANDIDATE_DIAG}/paired_predictions.csv").open("r", encoding="utf-8", newline="") as handle:
        candidate_rows = list(csv.DictReader(handle))
    if len(control_rows) != 29737 or len(candidate_rows) != 29737:
        raise ValueError("P12 diagnostics did not cover all 29,737 validation rows")
    count = wins = losses = control_correct = candidate_correct = 0
    for control, candidate in zip(control_rows, candidate_rows, strict=True):
        sample_id = control["sample_id"]
        if sample_id != candidate["sample_id"] or control["target"] != candidate["target"]:
            raise ValueError("P12 paired prediction rows do not align")
        if sample_id not in metadata:
            raise ValueError("P12 prediction sample missing from train manifest")
        sha, train_label = metadata[sample_id]
        if int(control["target"]) != int(train_label):
            raise ValueError("P12 target differs from Stage2 train manifest")
        if sha in conflict_hashes:
            continue
        count += 1
        control_ok = control["target"] == control["four_view_prediction"]
        candidate_ok = candidate["target"] == candidate["four_view_prediction"]
        control_correct += int(control_ok)
        candidate_correct += int(candidate_ok)
        wins += int(candidate_ok and not control_ok)
        losses += int(control_ok and not candidate_ok)
    if count == 0:
        raise ValueError("P12 ordinary sample group is empty")
    return {
        "count": count,
        "candidate_wins": wins,
        "control_wins": losses,
        "delta_accuracy_pp": (candidate_correct - control_correct) / count * 100.0,
        "definition": "exclude samples whose exact SHA appears in cross-label conflict audit",
    }


def _screen(checks: dict[str, object], reports: dict[str, object], audit: dict[str, object]) -> dict[str, object]:
    control, candidate = reports["control"], reports["candidate"]
    if any(r.get("scope") != "Stage 2 train-derived fixed validation only; no test access" for r in reports.values()):
        raise ValueError("P12 diagnostic scope is not train-only")
    if any(r.get("smoke_only") or r.get("evaluated_count") != 29737 for r in reports.values()):
        raise ValueError("P12 diagnostic is incomplete or smoke-only")
    if control.get("checkpoint_sha256") != checks["control"]["best_sha256"] or candidate.get("checkpoint_sha256") != checks["candidate"]["best_sha256"]:
        raise ValueError("P12 diagnostic checkpoint SHA mismatch")
    for name, report in (("control", control), ("candidate", candidate)):
        expected = checks[name]
        if report.get("experiment_config_sha256") != expected["config_sha256"]:
            raise ValueError(f"P12 {name} diagnostic config SHA mismatch")
        if report.get("manifest_sha256") != expected["manifest_sha256"] or report.get("split_sha256") != expected["split_sha256"]:
            raise ValueError(f"P12 {name} diagnostic data-source SHA mismatch")
        center_top1 = report.get("center", {}).get("top1")
        if center_top1 is None or abs(float(center_top1) - float(expected["best_top1"])) > 0.0002:
            raise ValueError(f"P12 {name} diagnostic center score does not match selected best checkpoint metric")
    if audit.get("baseline_column") != "four_view_prediction" or audit.get("candidate_column") != "four_view_prediction":
        raise ValueError("P12 bootstrap did not use the preregistered four-view predictions")
    if audit.get("replicates") != 500 or audit.get("count") != 29737:
        raise ValueError("P12 bootstrap sample count or replicate count mismatch")
    a = control["four_view"]
    b = candidate["four_view"]
    delta = {
        key: (float(b[key]) - float(a[key])) * 100.0
        for key in ("top1", "macro_accuracy", "tail_class_mean_recall")
    }
    grouped = audit["paired_near_duplicate_group_bootstrap"]
    ci_low_pp = float(grouped["ci95_low"]) * 100.0
    ordinary = _ordinary_delta()
    ac_decisions = {
        key: checks[key]["ac_decisions"] for key in ("control", "candidate")
    }
    ac_l1 = all(
        decisions and all(str(decision).upper() == "L1" for decision in decisions)
        for decisions in ac_decisions.values()
    )
    gate = {
        "four_view_top1_at_least_0_30pp": delta["top1"] >= 0.30,
        "near_duplicate_bootstrap_ci95_lower_gt_zero": ci_low_pp > 0.0,
        "macro_nonnegative": delta["macro_accuracy"] >= 0.0,
        "tail_at_least_minus_0_25pp": delta["tail_class_mean_recall"] >= -0.25,
        "ordinary_group_nonnegative": ordinary["delta_accuracy_pp"] >= 0.0,
        "both_ac_l1_all_epochs": ac_l1,
    }
    return {
        "schema": "aictrack1.p12_screen_result.v1",
        "queue_name": QUEUE_NAME,
        "status": "eligible_for_fold3_confirmation_only" if all(gate.values()) else "screen_failed_stop",
        "test_access": False,
        "control": {"run": CONTROL_RUN, "best_sha256": checks["control"]["best_sha256"], "four_view": a},
        "candidate": {"run": CANDIDATE_RUN, "best_sha256": checks["candidate"]["best_sha256"], "four_view": b},
        "delta_candidate_minus_control_pp": delta,
        "ordinary_sample_group": ordinary,
        "near_duplicate_group_bootstrap": {
            "replicates": grouped["replicates"], "groups": grouped["group_count"],
            "delta_pp": float(grouped["delta"]) * 100.0,
            "ci95_low_pp": ci_low_pp,
            "ci95_high_pp": float(grouped["ci95_high"]) * 100.0,
            "seed": 20260925,
        },
        "ac_decisions": ac_decisions,
        "gate": gate,
        "next_step": (
            "If all gates pass, preregister a matched fold3 confirmation; no full-train or test authorization is implied."
            if all(gate.values())
            else "Stop this ELR configuration; do not sweep parameters, seeds, full-train or test."
        ),
    }


def main() -> int:
    if QUEUE_DIR.exists() or any((ROOT / output).exists() for output in OUTPUTS):
        raise FileExistsError("P12 queue or output already exists; refusing implicit restart/overwrite")
    if any((ROOT / rel).exists() for rel in (CONTROL_DIAG, CANDIDATE_DIAG, BOOTSTRAP)):
        raise FileExistsError("P12 diagnostic output already exists; refusing overwrite")
    if read_training_lock(ROOT / DEFAULT_LOCK_RELATIVE_PATH) is not None:
        raise RuntimeError("repository training lock is not free")
    if probe_running_trainers():
        raise RuntimeError("a supported training process is already running")
    if not torch.cuda.is_available():
        raise RuntimeError("P12 requires local CUDA")
    free_gpu, total_gpu = torch.cuda.mem_get_info(0)
    disk_free = shutil.disk_usage(ROOT).free
    ram_free = _ram_available_bytes()
    if free_gpu < 5 * 1024**3 or disk_free < 12 * 1024**3 or ram_free < 8 * 1024**3:
        raise RuntimeError(
            f"P12 resources below frozen guards: free_gpu={free_gpu}, free_disk={disk_free}, free_ram={ram_free}"
        )
    _validate_configs()
    missing = [relative for relative in SOURCE_INPUTS if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"P12 frozen sources/inputs missing: {missing}")
    frozen_sha = {relative: _sha256(ROOT / relative) for relative in SOURCE_INPUTS}
    started = datetime.now(ZoneInfo("Asia/Shanghai"))
    deadline = started + timedelta(hours=HARD_BUDGET_HOURS)
    QUEUE_DIR.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": "aictrack1.p12_local_approval_receipt.v1",
        "queue_name": QUEUE_NAME,
        "scope": "local RTX4060; Stage2 fold2 train and train-derived validation only; matched 8-epoch SCE control then ELR candidate",
        "approval": "Continuing task is covered by the user's standing authorization for local-only training and full resource scheduling until the next compliant package; budget is explicitly bounded here.",
        "started_at": started.isoformat(),
        "hard_deadline": deadline.isoformat(),
        "expected_pair_wall_hours": [3.0, 4.5],
        "hard_budget_hours": HARD_BUDGET_HOURS,
        "expected_disk_delta_gib": [3.0, 5.0],
        "resource_free_before": {
            "gpu_bytes": free_gpu, "gpu_total_bytes": total_gpu,
            "disk_bytes": disk_free, "ram_bytes": ram_free,
        },
        "test_access": False,
        "remote_access": False,
        "source_input_sha256": frozen_sha,
    }
    receipt_path = QUEUE_DIR / "approval_receipt.json"
    _write_json(receipt_path, receipt)

    def verify_frozen() -> None:
        current = {relative: _sha256(ROOT / relative) for relative in SOURCE_INPUTS}
        if current != frozen_sha:
            changed = sorted(key for key in frozen_sha if frozen_sha[key] != current.get(key))
            raise RuntimeError(f"P12 frozen source/input SHA changed: {changed}")

    items = [
        QueueItem("sce_control", CONTROL, CONTROL_RUN, "matched fold2 SCE control under the same ELR-capable trainer"),
        QueueItem("elr_candidate", CANDIDATE, CANDIDATE_RUN, "single-variable basic ELR candidate"),
    ]

    def frozen_deadline_launcher(command, cwd, output, on_start):
        verify_frozen()
        # Start the exact queue-owned trainer once and record its PID in the shared lease.
        remaining = deadline.timestamp() - time.time()
        if remaining <= 0:
            return 124
        process = subprocess.Popen(list(command), cwd=cwd, stdout=output, stderr=subprocess.STDOUT)
        try:
            on_start(process.pid)
        except Exception:
            _taskkill_and_confirm(process)
            raise
        try:
            return process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            output.write("P12 hard deadline: stopping only this owned trainer process tree; latest atomic last.pt is retained.\n")
            output.flush()
            _taskkill_and_confirm(process)
            return 124

    queue_code = run_serial_queue(
        root=ROOT,
        python=Path(sys.executable),
        logs=QUEUE_DIR,
        queue_name=QUEUE_NAME,
        items=items,
        launcher=frozen_deadline_launcher,
    )
    if queue_code != 0:
        _write_json(QUEUE_DIR / "queue_status.json", {"status": "training_failed_or_incomplete", "exit_code": queue_code})
        return queue_code
    verify_frozen()
    checks = {
        "control": _validate_run(CONTROL_RUN, expect_elr=False, expected_run_id="20260929-stage2-p12-fold2-sce-control"),
        "candidate": _validate_run(CANDIDATE_RUN, expect_elr=True, expected_run_id="20260929-stage2-p12-fold2-elr-candidate"),
    }
    _write_json(QUEUE_DIR / "checkpoint_validation.json", checks)
    reports, audit = _run_locked_diagnostics(deadline.timestamp(), checks)
    verify_frozen()
    result = _screen(checks, reports, audit)
    result["source_input_sha256"] = frozen_sha
    result["approval_receipt_sha256"] = _sha256(receipt_path)
    result["diagnostic_artifacts"] = {
        CONTROL_DIAG: _sha256(ROOT / CONTROL_DIAG / "report.json"),
        f"{CONTROL_DIAG}/paired_predictions.csv": _sha256(ROOT / CONTROL_DIAG / "paired_predictions.csv"),
        CANDIDATE_DIAG: _sha256(ROOT / CANDIDATE_DIAG / "report.json"),
        f"{CANDIDATE_DIAG}/paired_predictions.csv": _sha256(ROOT / CANDIDATE_DIAG / "paired_predictions.csv"),
        BOOTSTRAP: _sha256(ROOT / BOOTSTRAP),
    }
    _write_json(QUEUE_DIR / "screen_result.json", result)
    _write_json(QUEUE_DIR / "queue_status.json", {"status": result["status"], "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()})
    print(json.dumps({"queue": QUEUE_NAME, "status": result["status"], "delta_top1_pp": result["delta_candidate_minus_control_pp"]["top1"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
