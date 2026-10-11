"""Run the approved P9 two-fold clip-norm screen and train-only diagnostics."""

from __future__ import annotations

import csv
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import QueueItem, run_parallel_queue  # noqa: E402


QUEUE_NAME = "20260928-p9-a10-clipnorm-pair"
LOGS = ROOT / "artifacts/queues" / QUEUE_NAME
CONTROL_CONFIGS = {
    "fold2": "configs/experiment/stage2_p6_sce_fold2_a10_ac_on_control.yaml",
    "fold3": "configs/experiment/stage2_p5_sce_fold3_a10_candidate.yaml",
}
CANDIDATE_CONFIGS = {
    "fold2": "configs/experiment/stage2_p9_fold2_sce_ac_clipnorm10_candidate.yaml",
    "fold3": "configs/experiment/stage2_p9_fold3_sce_ac_clipnorm10_candidate.yaml",
}
RUN_OUTPUTS = {
    "fold2": {
        "control": "artifacts/experiments/20260928-stage2-p6-fold2-a10-sce-ac-on-control",
        "candidate": "artifacts/experiments/20260928-stage2-p9-fold2-a10-sce-ac-clipnorm10-candidate",
    },
    "fold3": {
        "control": "artifacts/experiments/20260928-stage2-p5-fold3-a10-sce-candidate",
        "candidate": "artifacts/experiments/20260928-stage2-p9-fold3-a10-sce-ac-clipnorm10-candidate",
    },
}
BASELINE_BEST_SHA256 = {
    "fold2": "a59dfff6914f598702257d62759c8ace57131a8cf067c9e0f7161af9fca6bda8",
    "fold3": "2c550da87a7301d386f0b78cf4319635c8db9f3a65a07b4951047c53bb26f060",
}
EXPECTED_SHA256 = {
    "scripts/train_peft.py": "10f9eb4433cfb6d0c92be262c536363e4d6f2199eff840fb7a24feb974ef98ba",
    "src/aictrack1/engine/training_queue.py": "f3f2121e9a3e5dd63e4953d2732cd9f1f983d61cd8214e1c8b675e568ca3bfb7",
    "src/aictrack1/engine/peft.py": "449c85e68ced8968e5c54a022d05df83a3212eed9cf8870859d78008c82394ff",
    "src/aictrack1/engine/environment_identity.py": "b64c7bb998522c802b4bf2236c9dfb77cb6d45c633f493181f517f5ecb06f599",
    "src/aictrack1/engine/provenance.py": "6e30b38828a58f3bef055dc1dc651cc77404e16fa87b24edba242ca9213a71fb",
    "src/aictrack1/methods/sce.py": "5eb2da417cc47bc8f484d9cda03f453b3f1a079e9a8635c60ba6261ccb3687a9",
    "src/aictrack1/methods/confidence.py": "a1353e61fd3db49105067415d9d060d8be4ca0088d9e5fa37f56c6b71adf02f2",
    "src/aictrack1/models/clip_vit_b32.py": "6b025f2b3c2d3a3646552e1f9f9ce01912253709a27ef69a3cdf8b1a950eeffb",
    "src/aictrack1/models/clip_resolution.py": "ffcf84572405656e14f630a87e4c4a097bafc811351af8a4c7c23cef718a8090",
    "src/aictrack1/models/peft.py": "a3d5e87dd17714d44f69be8a9a8e71e2b18a9d30d22c9e3086922aac4b3a04fe",
    "src/aictrack1/models/heads.py": "f08294e44017f88182bae7e2d64029bb5ffc21d84f2a59a3f5c77bdc1884f525",
    "src/aictrack1/models/single_model.py": "6012fca832511af8cab126a49f28d6a2e3246978ac338d9e2fd0ca65f1e20839",
    "src/aictrack1/data/contracts.py": "922d1b049b7be9dc81767e02409a3f1b645bcfda74eb2bd6e74da5836a9482a3",
    "src/aictrack1/data/class_mapping.py": "c118b1b8b5ec4f60b6f8200d7274657b4605dd0eae8f46de74f3b6c2b8709f6c",
    "src/aictrack1/data/indexed_images.py": "49a5cda7640625f1fab15d3e20366855b0784890df1d3ef50148e5c1bca54e1f",
    "src/aictrack1/data/feature_cache.py": "59a93910bc242e3fa44da8cb11fb589b1120de18aa6b475e4ad5431ff5b273f0",
    "src/aictrack1/data/transforms.py": "99ca987dd1f1ee78e60616720c71af1ad7711d914f35c2b2974f64b24f6569d7",
    "src/aictrack1/data/manifest.py": "a61a6ccd22a2ce8c621b3e11f59da5fc40734c4afa4443650c400cd4efbf544c",
    "src/aictrack1/evaluation/classification.py": "3ae485c856cdbdce1a64d2be15339a9aac641ea323de971c385f7efc8b8324fa",
    "src/aictrack1/evaluation/long_tail.py": "3da2fec19cea59a94a6d50ac7fd63bba9f6052ece419a7086d392684846b43af",
    "src/aictrack1/evaluation/four_view.py": "8f1a6f305cedbfa0a96d276d6af61b9c4413efe358ce1cfdb0639530d2ba946b",
    "src/aictrack1/evaluation/feature_geometry.py": "fd34b54b0ae95d1ee3e42924e8e8f4e23e6b36abe0cb2d663bbd52acec48e193",
    "scripts/diagnose_stage2_tta.py": "ee3ebaf446fbe27f15147993be97e70e5bd2b243ae12ccc3c90f0dfcd586f8ce",
    "scripts/audit_stage2_paired_tta_gain.py": "3cc37d6d34302f991d951d9fc1fd89ce19c697e8cc48a338b7552178804f26b7",
    "configs/model/openai_clip_vit_b32.yaml": "9b1a6ca937486351383b5b437971b02bff914d772a4218009d0b4ed57e70f735",
    "configs/data/stage2_fold2_336.yaml": "a3736c041aa8fb1b2a5126a0842597f94d4b834bc84a41af90cc2768bf19cafe",
    "configs/data/stage2_fold3_336.yaml": "48d57f1de449ecadcd385b3224a39b792f07c5d1f2009f270f783f15391a6a86",
    "configs/experiment/stage2_p6_sce_fold2_a10_ac_on_control.yaml": "f6997daa8b2cff765c88d3a16c15a5c34ab4c4efe0d759069383b27199cb8b47",
    "configs/experiment/stage2_p5_sce_fold3_a10_candidate.yaml": "5b48db475a8c3b4ae315b2c5405eb1edba7fbe61f80111deeabb5948716aff5a",
    "configs/experiment/stage2_p9_fold2_sce_ac_clipnorm10_candidate.yaml": "fa45715a067e6f9ba8f25560b50b1ea014576271e4fdeaeb74b19ae16ea6a44d",
    "configs/experiment/stage2_p9_fold3_sce_ac_clipnorm10_candidate.yaml": "324f28d4aa35a6c7cee18665f47b885c6eef28bb4115a655150def4358ceaffa",
    "artifacts/audits/stage2/train/samples.csv": "1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d",
    "artifacts/audits/stage2/train/label_conflicts.csv": "b1afe084e654aeac5d431501657734c168a3cfc1f928defacac219dcf84c430a",
    "artifacts/audits/stage2/train/split_fold2_seeded_random_20260924.csv": "1e5eb7e9adbc234ba933bf76c99896602b8bdce5557879f90224ae2bb4cf2a50",
    "artifacts/audits/stage2/train/split_fold3_seeded_random_20260925.csv": "dcef8ccb919fec62d451fe78089d11399e7de2aaf779df7c636cdbdf176cdf68",
    "artifacts/experiments/20260925-stage2-fold2-c5-h1-336-seed20260921/best.pt": "83380620f06af442ddbe4e55f048d6519d03ce95e672ab3bc51469904b803500",
    "artifacts/experiments/20260925-stage2-fold3-c5-h1-336-seed20260921/best.pt": "1c41d274ff6a915e97065da515b5449242bdbf29c80a177976990c69bd7fd1df",
    "artifacts/experiments/20260928-stage2-p6-fold2-a10-sce-ac-on-control/best.pt": "a59dfff6914f598702257d62759c8ace57131a8cf067c9e0f7161af9fca6bda8",
    "artifacts/experiments/20260928-stage2-p6-fold2-a10-sce-ac-on-control/metrics.json": "3abbf1ffb6d1f5912f2cf4c4d3f675877f7352148d6aa3e99a5513731b2c60aa",
    "artifacts/experiments/20260928-stage2-p5-fold3-a10-sce-candidate/best.pt": "2c550da87a7301d386f0b78cf4319635c8db9f3a65a07b4951047c53bb26f060",
    "artifacts/experiments/20260928-stage2-p5-fold3-a10-sce-candidate/metrics.json": "4018c7e3eb957af917cdd3ec9922c511f22276f36122cd68c45129d4b707b677",
    "artifacts/weights/ViT-B-32.pt": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _check_frozen_inputs() -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, expected in EXPECTED_SHA256.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"frozen input missing: {relative}")
        actual[relative] = _sha256(path)
        if actual[relative] != expected:
            raise RuntimeError(f"frozen SHA mismatch for {relative}: {actual[relative]}")
    for fold, control_path in CONTROL_CONFIGS.items():
        control = yaml.safe_load((ROOT / control_path).read_text(encoding="utf-8"))["experiment"]
        candidate = yaml.safe_load(
            (ROOT / CANDIDATE_CONFIGS[fold]).read_text(encoding="utf-8")
        )["experiment"]
        changed = {
            key
            for key in control
            if key not in {"run_id", "output_dir"} and control[key] != candidate[key]
        }
        if changed != {"training"}:
            raise RuntimeError(f"{fold} candidate differs outside training: {sorted(changed)}")
        changed_training = {
            key
            for key in control["training"]
            if control["training"][key] != candidate["training"][key]
        }
        if changed_training != {"gradient_clip_norm"}:
            raise RuntimeError(f"{fold} training variables differ: {sorted(changed_training)}")
        if control["training"]["gradient_clip_norm"] != 1.0:
            raise RuntimeError(f"{fold} matched control clip norm is not 1.0")
        if candidate["training"]["gradient_clip_norm"] != 10.0:
            raise RuntimeError(f"{fold} candidate clip norm is not 10.0")
        if candidate.get("execution_enabled") is not True:
            raise RuntimeError(f"{fold} candidate has not been explicitly enabled")
    return actual


def _verify_checkpoint(path: Path, *, is_last: bool) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint is not a mapping: {path}")
    if is_last:
        required = {
            "epoch", "trainable_state", "optimizer", "scheduler", "scaler", "history",
            "python_rng_state", "numpy_rng_state", "torch_rng_state", "cuda_rng_state_all",
            "confidence_weighting",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"last checkpoint is missing resume state keys: {missing}")
    state = payload.get("trainable_state")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"checkpoint has no trainable_state: {path}")
    tensor_count = 0
    for value in state.values():
        if not torch.is_tensor(value):
            raise TypeError(f"non-tensor trainable state value: {path}")
        if value.is_floating_point() and not torch.isfinite(value).all().item():
            raise ValueError(f"non-finite trainable state tensor: {path}")
        tensor_count += 1
    return {"weights_only_load": True, "trainable_tensor_count": tensor_count, "epoch": payload.get("epoch")}


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)


def _diagnose(fold: str, arm: str) -> tuple[Path, Path, dict]:
    output = ROOT / "artifacts/diagnostics" / f"20260928-p9-{fold}-{arm}-clipnorm384-fourview"
    report_path = output / "report.json"
    predictions_path = output / "paired_predictions.csv"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"diagnostic output is not empty: {output}")
    run_output = ROOT / RUN_OUTPUTS[fold][arm]
    checkpoint = run_output / "best.pt"
    checkpoint_sha = _sha256(checkpoint)
    if arm == "control" and checkpoint_sha != BASELINE_BEST_SHA256[fold]:
        raise RuntimeError(f"{fold} control checkpoint SHA no longer matches frozen baseline")
    config = CONTROL_CONFIGS[fold] if arm == "control" else CANDIDATE_CONFIGS[fold]
    command = [
        sys.executable,
        "scripts/diagnose_stage2_tta.py",
        "--experiment-config", config,
        "--checkpoint", str(checkpoint),
        "--expected-checkpoint-sha256", checkpoint_sha,
        "--output-dir", str(output),
        "--batch-size", "64",
        "--workers", "4",
        "--four-view-scale384",
        "--save-fused-logits",
    ]
    _run(command, LOGS / f"diagnose-{fold}-{arm}.log")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("evaluated_count") != 29737 or report.get("smoke_only") is not False:
        raise ValueError(f"{fold}/{arm} diagnostic is not the full train validation split")
    if report.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError(f"{fold}/{arm} diagnostic checkpoint SHA mismatch")
    if report.get("four_view", {}).get("top1") is None:
        raise ValueError(f"{fold}/{arm} missing four-view metrics")
    if report.get("scope") != "Stage 2 train-derived fixed validation only; no test access":
        raise ValueError(f"{fold}/{arm} diagnostic scope is not the frozen train-only scope")
    metrics_path = run_output / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("schema") == "aictrack1.metrics_recovery.v1":
        history = metrics.get("history_summary") or []
        if not history:
            raise ValueError(f"{fold}/{arm} recovery metrics has no epoch history")
        best_row = max(history, key=lambda row: float(row["validation_top1"]))
        expected_center = {
            "top1": float(best_row["validation_top1"]),
            "macro_accuracy": float(best_row["validation_macro_accuracy"]),
            "tail_sample_accuracy": float(best_row["validation_tail_sample_accuracy"]),
        }
    else:
        history = metrics.get("history") or []
        if not history:
            raise ValueError(f"{fold}/{arm} native metrics has no epoch history")
        best_row = max(
            history,
            key=lambda row: float(row["validation"]["all"]["top1"]),
        )
        expected_center = {
            "top1": float(best_row["validation"]["all"]["top1"]),
            "macro_accuracy": float(best_row["validation"]["all"]["macro_accuracy"]),
            "tail_sample_accuracy": float(
                best_row["validation"]["long_tail"]["strata"]["tail"]["accuracy"]
            ),
        }
    center = report.get("center") or {}
    for metric, expected in expected_center.items():
        actual = center.get(metric)
        if actual is None or abs(float(actual) - expected) > 2e-6:
            raise ValueError(
                f"{fold}/{arm} center {metric} does not match its training metrics: "
                f"diagnostic={actual}, training={expected}"
            )
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)
    return report_path, predictions_path, report


def _ordinary_delta(fold: str, base_predictions: Path, candidate_predictions: Path) -> dict:
    split_name = (
        "split_fold2_seeded_random_20260924.csv"
        if fold == "fold2"
        else "split_fold3_seeded_random_20260925.csv"
    )
    split_path = ROOT / "artifacts/audits/stage2/train" / split_name
    validation_ids = set()
    with split_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "validation":
                validation_ids.add(row["sample_id"])
    sample_by_id = {}
    with (ROOT / "artifacts/audits/stage2/train/samples.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            if row["source_split"] != "train":
                raise ValueError("non-Stage2-train sample found in train manifest")
            if row["sample_id"] in validation_ids:
                sample_by_id[row["sample_id"]] = row
    conflict_hashes = set()
    with (ROOT / "artifacts/audits/stage2/train/label_conflicts.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        conflict_hashes = {row["sha256"] for row in csv.DictReader(handle)}

    def rows(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            data = list(csv.DictReader(handle))
        if len(data) != 29737:
            raise ValueError(f"expected 29,737 validation predictions: {path}")
        return data

    base, candidate = rows(base_predictions), rows(candidate_predictions)
    wins = losses = ordinary_count = base_correct = candidate_correct = 0
    for b, c in zip(base, candidate, strict=True):
        sample_id = b["sample_id"]
        if sample_id != c["sample_id"] or b["target"] != c["target"]:
            raise ValueError(f"paired prediction order mismatch in {fold}")
        sample = sample_by_id.get(sample_id)
        if sample is None:
            raise ValueError(f"prediction sample absent from Stage2 train manifest: {sample_id}")
        if sample["sha256"] in conflict_hashes:
            continue
        ordinary_count += 1
        b_ok = b["target"] == b["four_view_prediction"]
        c_ok = c["target"] == c["four_view_prediction"]
        base_correct += int(b_ok)
        candidate_correct += int(c_ok)
        wins += int(c_ok and not b_ok)
        losses += int(b_ok and not c_ok)
    if ordinary_count == 0:
        raise ValueError("ordinary subset unexpectedly empty")
    base_acc = base_correct / ordinary_count
    candidate_acc = candidate_correct / ordinary_count
    return {
        "ordinary_count": ordinary_count,
        "candidate_wins": wins,
        "control_wins": losses,
        "delta_accuracy_pp": (candidate_acc - base_acc) * 100.0,
    }


def _screen_result(diagnostics: dict, audits: dict, checkpoint_checks: dict) -> dict:
    folds = {}
    for fold in ("fold2", "fold3"):
        base = diagnostics[fold]["control"]["four_view"]
        cand = diagnostics[fold]["candidate"]["four_view"]
        delta = {
            key: (cand[key] - base[key]) * 100.0
            for key in ("top1", "macro_accuracy", "tail_class_mean_recall")
        }
        ordinary = _ordinary_delta(
            fold,
            diagnostics[fold]["control_predictions"],
            diagnostics[fold]["candidate_predictions"],
        )
        folds[fold] = {
            "control": base,
            "candidate": cand,
            "delta_pp": delta,
            "ordinary_sample_group": ordinary,
            "grouped_bootstrap": json.loads(audits[fold].read_text(encoding="utf-8"))[
                "paired_near_duplicate_group_bootstrap"
            ],
            "control_checkpoint_sha256": diagnostics[fold]["control_sha256"],
            "candidate_checkpoint_sha256": diagnostics[fold]["candidate_sha256"],
            "candidate_checkpoint_checks": checkpoint_checks[fold],
        }
    mean_top1 = sum(folds[fold]["delta_pp"]["top1"] for fold in folds) / 2.0
    mean_macro = sum(folds[fold]["delta_pp"]["macro_accuracy"] for fold in folds) / 2.0
    mean_tail = sum(folds[fold]["delta_pp"]["tail_class_mean_recall"] for fold in folds) / 2.0
    gate = {
        "average_top1_at_least_0_25pp": mean_top1 >= 0.25,
        "each_fold_top1_positive": all(folds[fold]["delta_pp"]["top1"] > 0.0 for fold in folds),
        "average_macro_nonnegative": mean_macro >= 0.0,
        "average_tail_at_least_minus_0_25pp": mean_tail >= -0.25,
        "each_fold_ordinary_group_nonnegative": all(
            folds[fold]["ordinary_sample_group"]["delta_accuracy_pp"] >= 0.0 for fold in folds
        ),
    }
    return {
        "schema": "aictrack1.p9_clipnorm_screen.v1",
        "scope": "Stage2 train-derived validation only; no test access",
        "candidate": "SCE(.1,1,-4)+AC, clip_norm=10; matched control clip_norm=1",
        "folds": folds,
        "mean_delta_pp": {"top1": mean_top1, "macro_accuracy": mean_macro, "tail_class_mean_recall": mean_tail},
        "gate": gate,
        "passed": all(gate.values()),
        "gate_note": "Exploration gate only; grouped bootstrap is reported descriptively and its CI crossing zero is not an independent veto under the user's relaxed exploration policy.",
        "next_step_if_passed": "request budget for multi-seed and 14-epoch matched confirmation; no full retrain or test inference is unlocked by this screen",
        "next_step_if_failed": "stop clip_norm=10 direction; do not scan another threshold",
    }


def main() -> int:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("P9 is approved only for the A10 Linux host")
    if LOGS.exists():
        raise RuntimeError(f"P9 queue directory already exists; no implicit restart: {LOGS}")
    disk_free = shutil.disk_usage(ROOT).free
    if disk_free < 10 * 1024**3:
        raise RuntimeError("P9 requires at least 10 GiB free before launch")
    if os.environ.get("AICTRACK1_P9_APPROVED") != "20260928-3to5h-hard6h-8GiB":
        raise RuntimeError("missing exact P9 approval environment receipt")
    source_hashes = _check_frozen_inputs()
    for fold in ("fold2", "fold3"):
        control_best = ROOT / RUN_OUTPUTS[fold]["control"] / "best.pt"
        if _sha256(control_best) != BASELINE_BEST_SHA256[fold]:
            raise RuntimeError(f"{fold} matched control best checkpoint failed SHA verification")
        if not (ROOT / RUN_OUTPUTS[fold]["control"] / "metrics.json").is_file():
            raise FileNotFoundError(f"{fold} control metrics missing")
    for paths in RUN_OUTPUTS.values():
        candidate_dir = ROOT / paths["candidate"]
        if candidate_dir.exists():
            raise FileExistsError(f"P9 candidate output already exists; no implicit restart: {candidate_dir}")

    started = datetime.now(ZoneInfo("Asia/Shanghai"))
    script_sha = _sha256(Path(__file__))
    receipt = {
        "schema": "aictrack1.p9_approval_receipt.v1",
        "queue_name": QUEUE_NAME,
        "approval": "User explicitly approved two-fold P9 screening: expected 3-5h A10, hard 6h, remote disk increase <=8GiB.",
        "started_at": started.isoformat(),
        "hard_deadline": (started + timedelta(hours=6)).isoformat(),
        "hard_deadline_enforcement": "external GNU timeout 6h TERM then KILL",
        "disk_free_bytes_before": disk_free,
        "expected_sha256": EXPECTED_SHA256,
        "actual_sha256": source_hashes,
        "queue_script_sha256": script_sha,
        "baseline_best_sha256": BASELINE_BEST_SHA256,
        "resource_budget": {"expected_a10_hours": "3-5", "hard_a10_hours": 6, "remote_disk_increment_gib_max": 8},
        "test_access": False,
    }
    LOGS.mkdir(parents=True, exist_ok=False)
    _write_json(LOGS / "approval_receipt.json", receipt)
    state = {"stage": "training", "started_at": started.isoformat(), "status": "running"}
    _write_json(LOGS / "p9_state.json", state)
    items = [
        QueueItem(
            tag="fold2_clipnorm10_candidate",
            config=CANDIDATE_CONFIGS["fold2"],
            output_dir=RUN_OUTPUTS["fold2"]["candidate"],
            purpose="P9 fold2 SCE+AC, only gradient clip norm 1->10",
        ),
        QueueItem(
            tag="fold3_clipnorm10_candidate",
            config=CANDIDATE_CONFIGS["fold3"],
            output_dir=RUN_OUTPUTS["fold3"]["candidate"],
            purpose="P9 fold3 SCE+AC, only gradient clip norm 1->10",
        ),
    ]
    train_rc = run_parallel_queue(
        root=ROOT,
        python=Path(sys.executable),
        logs=LOGS,
        queue_name=QUEUE_NAME,
        items=items,
        max_concurrent_jobs=2,
    )
    state["training_return_code"] = train_rc
    if train_rc != 0:
        state.update({"stage": "training", "status": "failed_or_incomplete"})
        _write_json(LOGS / "p9_state.json", state)
        return train_rc or 2

    state.update({"stage": "checkpoint_audit", "status": "running"})
    _write_json(LOGS / "p9_state.json", state)
    diagnostics = {"fold2": {}, "fold3": {}}
    checkpoint_checks = {}
    for fold in ("fold2", "fold3"):
        candidate_dir = ROOT / RUN_OUTPUTS[fold]["candidate"]
        metrics = json.loads((candidate_dir / "metrics.json").read_text(encoding="utf-8"))
        expected_epochs = int(
            yaml.safe_load((ROOT / CANDIDATE_CONFIGS[fold]).read_text(encoding="utf-8"))[
                "experiment"
            ]["training"]["epochs"]
        )
        if (
            metrics.get("training_scope") != "fixed_split_validation"
            or int(metrics.get("train_count", -1)) != 118958
            or int(metrics.get("validation_count", -1)) != 29737
            or len(history := (metrics.get("history") or [])) != expected_epochs
            or int(history[-1].get("epoch", -1)) != expected_epochs
            or metrics.get("early_stop") is not None
        ):
            raise RuntimeError(f"{fold} candidate did not complete the configured full screening run")
        if metrics.get("git_commit") not in {"unknown_archive"} and not metrics.get("git_commit"):
            raise RuntimeError(f"{fold} source provenance is empty")
        decisions = []
        for row in history:
            effective = (row.get("confidence_weighting") or {}).get("next_effective_quality") or {}
            decisions.append(effective.get("decision"))
        if (
            any(decision is None or decision == "L2" for decision in decisions)
            or any(row.get("early_stop") for row in history)
        ):
            raise RuntimeError(f"{fold} candidate AC effective quality triggered L2")
        best = candidate_dir / "best.pt"
        last = candidate_dir / "last.pt"
        expected_best = str(metrics.get("best_checkpoint_sha256") or metrics.get("selected_checkpoint_sha256"))
        if _sha256(best) != expected_best:
            raise RuntimeError(f"{fold} best checkpoint SHA does not match metrics")
        checkpoint_checks[fold] = {
            "best": _verify_checkpoint(best, is_last=False),
            "best_sha256": _sha256(best),
            "last": _verify_checkpoint(last, is_last=True),
            "last_sha256": _sha256(last),
            "config_sha256": _sha256(ROOT / CANDIDATE_CONFIGS[fold]),
            "metrics_sha256": _sha256(candidate_dir / "metrics.json"),
            "ac_decisions": decisions,
        }
        diagnostics[fold]["candidate_sha256"] = _sha256(best)

    state.update({"stage": "four_view_validation", "status": "running"})
    _write_json(LOGS / "p9_state.json", state)
    diagnostic_report_paths = {}
    for fold in ("fold2", "fold3"):
        for arm in ("control", "candidate"):
            report_path, predictions_path, report = _diagnose(fold, arm)
            diagnostics[fold][arm] = report
            diagnostics[fold][f"{arm}_predictions"] = predictions_path
            diagnostics[fold][f"{arm}_sha256"] = report["checkpoint_sha256"]
            diagnostic_report_paths[f"{fold}_{arm}"] = {
                "report": str(report_path.relative_to(ROOT)),
                "report_sha256": _sha256(report_path),
                "predictions": str(predictions_path.relative_to(ROOT)),
                "predictions_sha256": _sha256(predictions_path),
            }

    state.update({"stage": "paired_group_bootstrap", "status": "running"})
    _write_json(LOGS / "p9_state.json", state)
    audit_paths = {}
    for fold in ("fold2", "fold3"):
        output = ROOT / "artifacts/reports" / f"20260928-p9-{fold}-clipnorm10-vs1-bootstrap500.json"
        command = [
            sys.executable,
            "scripts/audit_stage2_paired_tta_gain.py",
            "--baseline-predictions", str(diagnostics[fold]["control_predictions"]),
            "--candidate-predictions", str(diagnostics[fold]["candidate_predictions"]),
            "--baseline-column", "four_view_prediction",
            "--candidate-column", "four_view_prediction",
            "--output", str(output),
            "--seed", "20260925",
            "--replicates", "500",
        ]
        _run(command, LOGS / f"bootstrap-{fold}.log")
        audit_paths[fold] = output

    result = _screen_result(diagnostics, audit_paths, checkpoint_checks)
    result.update({
        "approval_receipt_sha256": _sha256(LOGS / "approval_receipt.json"),
        "source_sha256": source_hashes,
        "diagnostic_artifacts": diagnostic_report_paths,
        "bootstrap_reports": {
            fold: {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for fold, path in audit_paths.items()
        },
    })
    _write_json(LOGS / "screen_result.json", result)
    state.update({"stage": "complete", "status": "screen_passed" if result["passed"] else "screen_failed", "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()})
    _write_json(LOGS / "p9_state.json", state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
