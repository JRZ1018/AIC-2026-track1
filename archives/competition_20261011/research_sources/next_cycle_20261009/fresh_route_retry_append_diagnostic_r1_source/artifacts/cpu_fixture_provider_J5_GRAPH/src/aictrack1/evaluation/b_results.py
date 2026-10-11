"""Strictly validate and aggregate paired B-worker PEFT results."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import yaml


REQUIRED_RESULT_FILES = ("config.yaml", "command.txt", "metrics.json", "best.pt", "last.pt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_batch_plan(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    plan = document.get("batch", document)
    required = {
        "name",
        "manifest_sha256",
        "split_sha256",
        "train_count",
        "validation_count",
        "h1_reference_top1",
        "minimum_m1_mean_gain",
        "runs",
    }
    missing = required - set(plan)
    if missing:
        raise ValueError(f"B-result plan is missing fields: {sorted(missing)}")
    runs = plan["runs"]
    if not isinstance(runs, list) or not runs:
        raise ValueError("B-result plan must contain a non-empty runs list")
    identities = {(str(run["pair_id"]), str(run["role"])) for run in runs}
    if len(identities) != len(runs):
        raise ValueError("B-result plan contains duplicate pair/role entries")
    return plan


def check_readiness(plan_path: Path, root: Path) -> dict[str, Any]:
    plan = load_batch_plan(plan_path)
    missing: list[str] = []
    ready_runs: list[str] = []
    for spec in plan["runs"]:
        result_dir = root / str(spec["result_dir"])
        absent = [name for name in REQUIRED_RESULT_FILES if not (result_dir / name).is_file()]
        if result_dir.is_dir() and not absent:
            ready_runs.append(str(spec["run_id"]))
        else:
            missing.extend(str(result_dir / name) for name in absent)
    return {
        "batch": str(plan["name"]),
        "ready": not missing,
        "ready_runs": ready_runs,
        "missing_files": missing,
    }


def _read_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    value = value.get("experiment", value)
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def _best_validation(history: list[dict[str, Any]]) -> tuple[int, float, float]:
    if not history:
        raise ValueError("metrics history is empty")
    candidates: list[tuple[int, float, float]] = []
    for row in history:
        validation = row["validation"]["all"]
        epoch = int(row["epoch"])
        top1 = float(validation["top1"])
        macro = float(validation["macro_accuracy"])
        if not math.isfinite(top1) or not math.isfinite(macro):
            raise ValueError("metrics contain a non-finite validation value")
        candidates.append((epoch, top1, macro))
    return max(candidates, key=lambda item: item[1])


def inspect_run(spec: dict[str, Any], plan: dict[str, Any], root: Path) -> dict[str, Any]:
    run_id = str(spec["run_id"])
    expected_config_path = root / str(spec["config"])
    result_dir = root / str(spec["result_dir"])
    for name in REQUIRED_RESULT_FILES:
        if not (result_dir / name).is_file():
            raise FileNotFoundError(f"{run_id} is missing {name}: {result_dir / name}")

    expected_config = _read_mapping(expected_config_path)
    returned_config_path = result_dir / "config.yaml"
    returned_config = _read_mapping(returned_config_path)
    returned_config_sha256 = sha256_file(returned_config_path)
    canonical_config_sha256 = canonical_text_sha256(expected_config_path)
    if canonical_text_sha256(returned_config_path) != canonical_config_sha256:
        raise ValueError(
            f"{run_id} returned config differs from the locked repository config "
            "beyond LF/CRLF normalization"
        )
    if returned_config != expected_config:
        raise ValueError(f"{run_id} returned config has a semantic mismatch")
    if str(expected_config.get("run_id")) != run_id:
        raise ValueError(f"{run_id} does not match its config run_id")
    configured_seed = int(expected_config["training"]["seed"])
    if configured_seed != int(spec["seed"]):
        raise ValueError(f"{run_id} seed differs from the batch plan")
    if Path(str(expected_config["output_dir"])).name != Path(str(spec["result_dir"])).name:
        raise ValueError(f"{run_id} output directory differs from the locked config")

    metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    exact_checks = {
        "run_id": run_id,
        "config_sha256": returned_config_sha256,
        "manifest_sha256": str(plan["manifest_sha256"]),
        "split_sha256": str(plan["split_sha256"]),
        "git_commit": str(spec["expected_git_commit"]),
        "train_count": int(plan["train_count"]),
        "validation_count": int(plan["validation_count"]),
    }
    for field, expected in exact_checks.items():
        if metrics.get(field) != expected:
            raise ValueError(
                f"{run_id} metrics {field} mismatch: {metrics.get(field)!r} != {expected!r}"
            )
    if "test data not accessed" not in str(metrics.get("scope", "")).lower():
        raise ValueError(f"{run_id} does not attest that test data was not accessed")

    command = (result_dir / "command.txt").read_text(encoding="utf-8").strip()
    normalized_command = command.replace("\\", "/")
    normalized_config = str(spec["config"]).replace("\\", "/")
    if "train_peft.py" not in normalized_command or normalized_config not in normalized_command:
        raise ValueError(f"{run_id} command does not use its locked PEFT config")
    forbidden = ("infer_and_build_submission", "datasetstage1/test", "datasetstage1\\test")
    if any(token in command.lower() for token in forbidden):
        raise ValueError(f"{run_id} command contains a forbidden test/submission path")

    checkpoint_hashes = {
        "best_checkpoint_sha256": sha256_file(result_dir / "best.pt"),
        "last_checkpoint_sha256": sha256_file(result_dir / "last.pt"),
    }
    for field, actual in checkpoint_hashes.items():
        if metrics.get(field) != actual:
            raise ValueError(f"{run_id} {field} does not match the returned checkpoint")

    best_epoch, best_top1, best_macro = _best_validation(metrics["history"])
    if not math.isclose(float(metrics["best_top1"]), best_top1, abs_tol=1e-12):
        raise ValueError(f"{run_id} best_top1 does not match its validation history")
    return {
        "run_id": run_id,
        "pair_id": str(spec["pair_id"]),
        "role": str(spec["role"]),
        "seed": configured_seed,
        "git_commit": str(metrics["git_commit"]),
        "config_sha256": returned_config_sha256,
        "canonical_config_sha256": canonical_config_sha256,
        "best_epoch": best_epoch,
        "top1": best_top1,
        "macro_accuracy": best_macro,
        "duration_seconds": float(metrics["duration_seconds"]),
        "peak_vram_mib": float(metrics["peak_vram_mib"]),
        **checkpoint_hashes,
        "result_dir": str(spec["result_dir"]),
    }


def aggregate_batch(plan_path: Path, root: Path) -> dict[str, Any]:
    plan = load_batch_plan(plan_path)
    runs = [inspect_run(spec, plan, root) for spec in plan["runs"]]
    pairs: list[dict[str, Any]] = []
    for pair_id in sorted({str(run["pair_id"]) for run in runs}):
        members = {run["role"]: run for run in runs if run["pair_id"] == pair_id}
        if set(members) != {"m0", "m1"}:
            raise ValueError(f"pair {pair_id} must contain exactly one m0 and one m1 run")
        if members["m0"]["seed"] != members["m1"]["seed"]:
            raise ValueError(f"pair {pair_id} does not use a shared seed")
        pairs.append(
            {
                "pair_id": pair_id,
                "seed": members["m0"]["seed"],
                "m0_top1": members["m0"]["top1"],
                "m1_top1": members["m1"]["top1"],
                "m1_minus_m0": members["m1"]["top1"] - members["m0"]["top1"],
            }
        )

    by_role = {role: [run for run in runs if run["role"] == role] for role in ("m0", "m1")}
    summary: dict[str, Any] = {}
    for role, role_runs in by_role.items():
        values = [float(run["top1"]) for run in role_runs]
        macros = [float(run["macro_accuracy"]) for run in role_runs]
        summary[role] = {
            "count": len(values),
            "mean_top1": statistics.fmean(values),
            "population_std_top1": statistics.pstdev(values),
            "minimum_top1": min(values),
            "mean_macro_accuracy": statistics.fmean(macros),
        }

    gains = [float(pair["m1_minus_m0"]) for pair in pairs]
    mean_gain = statistics.fmean(gains)
    all_positive = all(gain > 0 for gain in gains)
    stable = all_positive and mean_gain >= float(plan["minimum_m1_mean_gain"])
    h1_reference = float(plan["h1_reference_top1"])
    if not stable:
        decision = "stop_m1_unstable_or_too_small"
    elif summary["m1"]["mean_top1"] <= h1_reference:
        decision = "retain_m1_mechanism_only_h1_remains_stronger"
    else:
        decision = "m1_is_stable_and_exceeds_h1_reference"

    return {
        "batch": str(plan["name"]),
        "scope": "train/validation result acceptance only; test data and submissions not accessed",
        "runs": runs,
        "pairs": pairs,
        "summary": summary,
        "decision_gate": {
            "minimum_m1_mean_gain": float(plan["minimum_m1_mean_gain"]),
            "mean_paired_gain": mean_gain,
            "all_pair_gains_positive": all_positive,
            "m1_stable_over_m0": stable,
            "h1_reference_top1": h1_reference,
            "decision": decision,
        },
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite B-result report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
