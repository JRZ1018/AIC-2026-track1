"""Canonical builder and validator for the central experiment index."""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .environment_identity import UNKNOWN_LEGACY, sha256_file, verified_environment_hash


INDEX_FIELDS = (
    "run_id",
    "stage",
    "method",
    "upstream_candidates",
    "seed",
    "data_hash",
    "split_hash",
    "config_hash",
    "environment_hash",
    "git_commit",
    "status",
    "top1",
    "macro_accuracy",
    "head_accuracy",
    "mid_accuracy",
    "tail_accuracy",
    "peak_vram_mib",
    "duration_seconds",
    "best_checkpoint_sha256",
    "last_checkpoint_sha256",
    "result_dir",
    "started_at",
    "finished_at",
    "decision",
    "notes",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_LEGACY_NOTE = "environment unknown_legacy: complete platform/lock/model identity was not recorded"


@dataclass(frozen=True)
class IndexAudit:
    original_rows: int
    final_rows: int
    added_rows: int
    hashed_environments: int
    unknown_legacy_environments: int
    restored_ledger_fields: int
    refreshed_checkpoint_fields: int
    verified_checkpoints: int


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def read_index(path: str | Path) -> list[dict[str, str]]:
    index_path = Path(path)
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != INDEX_FIELDS:
            raise ValueError("experiment index header does not match the canonical schema")
        rows = [{field: str(row.get(field, "")) for field in INDEX_FIELDS} for row in reader]
    run_ids = [row["run_id"] for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("experiment index contains duplicate run_id values")
    return rows


def read_index_text(text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    if tuple(reader.fieldnames or ()) != INDEX_FIELDS:
        raise ValueError("experiment index header does not match the canonical schema")
    rows = [{field: str(row.get(field, "")) for field in INDEX_FIELDS} for row in reader]
    run_ids = [row["run_id"] for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("experiment index contains duplicate run_id values")
    return rows


def write_index(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    index_path = Path(path)
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in INDEX_FIELDS})
    temporary.replace(index_path)


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    raw = payload.get("experiment", payload)
    if not isinstance(raw, dict):
        raise ValueError(f"experiment configuration must be a mapping: {path}")
    return raw


def _method(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    peft = metrics.get("peft", config.get("peft"))
    if isinstance(peft, Mapping) and peft.get("mode"):
        return f"peft_{peft['mode']}"
    loss = metrics.get("loss")
    if isinstance(loss, Mapping) and loss.get("name"):
        return f"frozen_head_{loss['name']}"
    return "training_run"


def _stage(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    provenance = metrics.get("checkpoint_provenance")
    if isinstance(provenance, Mapping) and provenance.get("competition_stage"):
        return str(provenance["competition_stage"])
    if config.get("competition_stage"):
        return str(config["competition_stage"])
    return "preliminary"


def _seed(config: Mapping[str, Any]) -> str:
    training = config.get("training")
    if isinstance(training, Mapping) and training.get("seed") is not None:
        return str(training["seed"])
    if config.get("seed") is not None:
        return str(config["seed"])
    return ""


def _checkpoint_hash(metrics: Mapping[str, Any]) -> str:
    for key in (
        "selected_checkpoint_sha256",
        "best_checkpoint_sha256",
        "final_checkpoint_sha256",
    ):
        value = metrics.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _append_note(notes: str, addition: str) -> str:
    if addition in notes:
        return notes
    return f"{notes}; {addition}" if notes else addition


def _scalar(value: Any) -> str:
    return "" if value is None else str(value)


def _resolve_within(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"experiment result_dir escapes repository: {relative!r}") from exc
    return candidate


def row_from_result(root: Path, result_dir: Path) -> dict[str, str]:
    metrics_path = result_dir / "metrics.json"
    metrics = _read_json(metrics_path)
    config_path = result_dir / "config.yaml"
    config = _load_config(config_path)
    run_id = str(metrics.get("run_id") or config.get("run_id") or result_dir.name)
    if run_id != result_dir.name:
        raise ValueError(f"run_id/result directory mismatch: {run_id!r} != {result_dir.name!r}")
    env_hash = verified_environment_hash(metrics) or UNKNOWN_LEGACY
    notes = "Automatically indexed from metrics.json"
    if env_hash == UNKNOWN_LEGACY:
        notes = _append_note(notes, _LEGACY_NOTE)
    config_hash = str(metrics.get("config_sha256") or "")
    if not config_hash and config_path.is_file():
        config_hash = sha256_file(config_path)
    relative = result_dir.relative_to(root).as_posix()
    top1 = metrics.get("best_top1")
    return {
        "run_id": run_id,
        "stage": _stage(metrics, config),
        "method": _method(metrics, config),
        "upstream_candidates": "openai_clip",
        "seed": _seed(config),
        "data_hash": str(metrics.get("dataset_config_sha256") or ""),
        "split_hash": str(metrics.get("split_sha256") or ""),
        "config_hash": config_hash,
        "environment_hash": env_hash,
        "git_commit": str(metrics.get("git_commit") or ""),
        "status": "completed",
        "top1": "" if top1 is None else str(top1),
        "macro_accuracy": "",
        "head_accuracy": "",
        "mid_accuracy": "",
        "tail_accuracy": "",
        "peak_vram_mib": _scalar(metrics.get("peak_vram_mib")),
        "duration_seconds": _scalar(metrics.get("duration_seconds")),
        "best_checkpoint_sha256": _checkpoint_hash(metrics),
        "last_checkpoint_sha256": str(metrics.get("last_checkpoint_sha256") or ""),
        "result_dir": relative,
        "started_at": "",
        "finished_at": "",
        "decision": "unreviewed",
        "notes": notes,
    }


def _modern_hash_for_result(result_dir: Path) -> str | None:
    metrics_path = result_dir / "metrics.json"
    if metrics_path.is_file():
        value = verified_environment_hash(_read_json(metrics_path))
        if value:
            return value
    environment_path = result_dir / "environment.json"
    if environment_path.is_file():
        return verified_environment_hash(_read_json(environment_path))
    return None


def _refresh_reproducibility_fields(row: dict[str, str], result_dir: Path) -> int:
    """Refresh machine-derived fields while preserving human decisions and notes."""

    metrics_path = result_dir / "metrics.json"
    if not metrics_path.is_file():
        return 0
    metrics = _read_json(metrics_path)
    updates = {
        "best_checkpoint_sha256": _checkpoint_hash(metrics),
        "last_checkpoint_sha256": str(metrics.get("last_checkpoint_sha256") or ""),
    }
    changed_checkpoints = 0
    for field, value in updates.items():
        if not value:
            continue
        if field in {"best_checkpoint_sha256", "last_checkpoint_sha256"} and row[field] != value:
            changed_checkpoints += 1
        row[field] = value
    return changed_checkpoints


def _validate_digest(value: str, context: str) -> None:
    if value and _SHA256.fullmatch(value) is None:
        raise ValueError(f"{context} is not a SHA-256 digest: {value!r}")


def verify_row_checkpoints(root: Path, row: Mapping[str, str]) -> int:
    result_dir = _resolve_within(root, row["result_dir"])
    metrics_path = result_dir / "metrics.json"
    if not metrics_path.is_file():
        return 0
    metrics = _read_json(metrics_path)
    verified = 0
    selected_hash = row["best_checkpoint_sha256"]
    _validate_digest(selected_hash, f"{row['run_id']} selected checkpoint hash")
    selected_name = metrics.get("selected_checkpoint")
    if selected_hash and isinstance(selected_name, str):
        selected_path = result_dir / selected_name
        if not selected_path.is_file():
            raise FileNotFoundError(f"selected checkpoint is missing: {selected_path}")
        actual = sha256_file(selected_path)
        if actual != selected_hash:
            raise ValueError(f"{row['run_id']} selected checkpoint hash mismatch")
        verified += 1
    last_hash = row["last_checkpoint_sha256"]
    _validate_digest(last_hash, f"{row['run_id']} last checkpoint hash")
    if last_hash:
        last_path = result_dir / "last.pt"
        if not last_path.is_file():
            raise FileNotFoundError(f"last checkpoint is missing: {last_path}")
        if sha256_file(last_path) != last_hash:
            raise ValueError(f"{row['run_id']} last checkpoint hash mismatch")
        verified += 1
    return verified


def rebuild_index(
    root: str | Path,
    index_path: str | Path,
    *,
    verify_checkpoints: bool = False,
    baseline_rows: Iterable[Mapping[str, str]] | None = None,
    allow_legacy_discovery: bool = False,
) -> tuple[list[dict[str, str]], IndexAudit]:
    root_path = Path(root).resolve()
    index_file = Path(index_path).resolve()
    rows = read_index(index_file)
    original_rows = len(rows)
    restored_ledger_fields = 0
    if baseline_rows is not None:
        baseline_by_id = {str(row["run_id"]): row for row in baseline_rows}
        for row in rows:
            baseline = baseline_by_id.get(row["run_id"])
            if baseline is None:
                continue
            for field in ("data_hash", "split_hash", "config_hash", "git_commit"):
                baseline_value = str(baseline.get(field, ""))
                if baseline_value and row[field] != baseline_value:
                    row[field] = baseline_value
                    restored_ledger_fields += 1
    by_run_id = {row["run_id"]: row for row in rows}
    refreshed_checkpoint_fields = 0
    for row in rows:
        result_dir = _resolve_within(root_path, row["result_dir"])
        refreshed_checkpoint_fields += _refresh_reproducibility_fields(row, result_dir)
        modern_hash = _modern_hash_for_result(result_dir)
        row["environment_hash"] = modern_hash or UNKNOWN_LEGACY
        if modern_hash is None:
            row["notes"] = _append_note(row["notes"], _LEGACY_NOTE)
    experiment_root = root_path / "artifacts" / "experiments"
    if experiment_root.is_dir():
        for result_dir in sorted(experiment_root.iterdir(), key=lambda path: path.name):
            if not result_dir.is_dir() or not (result_dir / "metrics.json").is_file():
                continue
            candidate = row_from_result(root_path, result_dir)
            if candidate["run_id"] not in by_run_id:
                if (
                    candidate["environment_hash"] == UNKNOWN_LEGACY
                    and not allow_legacy_discovery
                ):
                    raise ValueError(
                        f"new run {candidate['run_id']!r} lacks a complete environment identity"
                    )
                rows.append(candidate)
                by_run_id[candidate["run_id"]] = candidate
    if len(by_run_id) != len(rows):
        raise ValueError("rebuilt experiment index contains duplicate run_id values")
    verified = 0
    if verify_checkpoints:
        for row in rows:
            verified += verify_row_checkpoints(root_path, row)
    hashed = sum(1 for row in rows if _SHA256.fullmatch(row["environment_hash"]) is not None)
    unknown = sum(1 for row in rows if row["environment_hash"] == UNKNOWN_LEGACY)
    if hashed + unknown != len(rows):
        raise ValueError("every experiment must have a verified hash or unknown_legacy")
    return rows, IndexAudit(
        original_rows=original_rows,
        final_rows=len(rows),
        added_rows=len(rows) - original_rows,
        hashed_environments=hashed,
        unknown_legacy_environments=unknown,
        restored_ledger_fields=restored_ledger_fields,
        refreshed_checkpoint_fields=refreshed_checkpoint_fields,
        verified_checkpoints=verified,
    )
