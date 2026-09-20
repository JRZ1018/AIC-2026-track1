from __future__ import annotations

import json
from pathlib import Path

import pytest

from aictrack1.engine.environment_identity import ENVIRONMENT_SCHEMA, environment_hash, sha256_file
from aictrack1.engine.experiment_index import (
    INDEX_FIELDS,
    read_index,
    read_index_text,
    rebuild_index,
    write_index,
)


def _identity() -> dict[str, object]:
    return {
        "schema": ENVIRONMENT_SCHEMA,
        "python": {"implementation": "CPython", "version": "3.12.10"},
        "platform": {
            "system": "Windows",
            "release": "11",
            "version": "10.0.26100",
            "machine": "AMD64",
        },
        "pytorch": {"version": "2.11.0+cu128", "cuda_runtime": "12.8", "cudnn": 91002},
        "accelerator": {"device": "cuda", "gpu": "Synthetic GPU"},
        "dependencies": {
            "requirements_lock_path": "requirements-lock.txt",
            "requirements_lock_sha256": "a" * 64,
        },
        "model": {
            "provider": "OpenAI",
            "name": "ViT-B/32",
            "clip_weight_sha256": "b" * 64,
        },
    }


def _blank_row(run_id: str, result_dir: str) -> dict[str, str]:
    row = {field: "" for field in INDEX_FIELDS}
    row.update(
        run_id=run_id,
        stage="preliminary",
        method="legacy",
        status="completed",
        result_dir=result_dir,
        decision="keep",
        notes="human note",
    )
    return row


def _write_modern_run(root: Path, run_id: str = "modern-run") -> Path:
    result = root / "artifacts" / "experiments" / run_id
    result.mkdir(parents=True)
    selected = result / "final.pt"
    last = result / "last.pt"
    selected.write_bytes(b"selected")
    last.write_bytes(b"last")
    identity = _identity()
    metrics = {
        "run_id": run_id,
        "best_top1": None,
        "duration_seconds": 12.5,
        "peak_vram_mib": 0,
        "selected_checkpoint": "final.pt",
        "selected_checkpoint_sha256": sha256_file(selected),
        "last_checkpoint_sha256": sha256_file(last),
        "dataset_config_sha256": "c" * 64,
        "split_sha256": "d" * 64,
        "git_commit": "e" * 40,
        "peft": {"mode": "last_n_blocks"},
        "environment": identity,
        "environment_hash": environment_hash(identity),
    }
    (result / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (result / "config.yaml").write_text(
        f"experiment:\n  run_id: {run_id}\n  training:\n    seed: 7\n",
        encoding="utf-8",
    )
    return result


def test_rebuild_backfills_legacy_adds_run_and_is_idempotent(tmp_path: Path) -> None:
    index = tmp_path / "experiments" / "index.csv"
    index.parent.mkdir(parents=True)
    legacy = tmp_path / "artifacts" / "experiments" / "legacy-run"
    legacy.mkdir(parents=True)
    (legacy / "metrics.json").write_text(json.dumps({"best_top1": 0.5}), encoding="utf-8")
    write_index(index, [_blank_row("legacy-run", "artifacts/experiments/legacy-run")])
    _write_modern_run(tmp_path)
    duplicate_return = tmp_path / "artifacts" / "returns" / "copy" / "modern-run"
    duplicate_return.mkdir(parents=True)
    (duplicate_return / "metrics.json").write_text("{}", encoding="utf-8")

    rows, audit = rebuild_index(tmp_path, index, verify_checkpoints=True)
    assert audit.original_rows == 1
    assert audit.final_rows == 2
    assert audit.added_rows == 1
    assert audit.hashed_environments == 1
    assert audit.unknown_legacy_environments == 1
    assert audit.refreshed_checkpoint_fields == 0
    assert audit.verified_checkpoints == 2
    by_id = {row["run_id"]: row for row in rows}
    assert by_id["legacy-run"]["environment_hash"] == "unknown_legacy"
    assert "human note" in by_id["legacy-run"]["notes"]
    assert "unknown_legacy" in by_id["legacy-run"]["notes"]
    assert by_id["modern-run"]["method"] == "peft_last_n_blocks"
    assert by_id["modern-run"]["seed"] == "7"
    assert by_id["modern-run"]["peak_vram_mib"] == "0"

    write_index(index, rows)
    rebuilt, second = rebuild_index(tmp_path, index)
    assert second.added_rows == 0
    assert rebuilt == read_index(index)


def test_rebuild_repairs_machine_derived_checkpoint_columns(tmp_path: Path) -> None:
    index = tmp_path / "experiments" / "index.csv"
    index.parent.mkdir(parents=True)
    result = _write_modern_run(tmp_path, "repair-run")
    row = _blank_row("repair-run", "artifacts/experiments/repair-run")
    row["last_checkpoint_sha256"] = sha256_file(result / "final.pt")
    write_index(index, [row])

    rows, audit = rebuild_index(tmp_path, index, verify_checkpoints=True)
    metrics = json.loads((result / "metrics.json").read_text(encoding="utf-8"))
    assert rows[0]["best_checkpoint_sha256"] == metrics["selected_checkpoint_sha256"]
    assert rows[0]["last_checkpoint_sha256"] == metrics["last_checkpoint_sha256"]
    assert audit.refreshed_checkpoint_fields == 2


def test_rebuild_can_restore_established_ledger_fields(tmp_path: Path) -> None:
    index = tmp_path / "experiments" / "index.csv"
    index.parent.mkdir(parents=True)
    result = _write_modern_run(tmp_path, "preserve-run")
    current = _blank_row("preserve-run", "artifacts/experiments/preserve-run")
    current["data_hash"] = "1" * 64
    current["git_commit"] = "1" * 40
    baseline = dict(current)
    baseline["data_hash"] = "2" * 64
    baseline["git_commit"] = "2" * 40
    write_index(index, [current])

    rows, audit = rebuild_index(tmp_path, index, baseline_rows=[baseline])
    assert rows[0]["data_hash"] == "2" * 64
    assert rows[0]["git_commit"] == "2" * 40
    assert audit.restored_ledger_fields == 2
    assert rows[0]["best_checkpoint_sha256"] == sha256_file(result / "final.pt")


def test_rebuild_rejects_checkpoint_mismatch(tmp_path: Path) -> None:
    index = tmp_path / "experiments" / "index.csv"
    index.parent.mkdir(parents=True)
    write_index(index, [])
    result = _write_modern_run(tmp_path)
    (result / "final.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="selected checkpoint hash mismatch"):
        rebuild_index(tmp_path, index, verify_checkpoints=True)


def test_rebuild_rejects_result_dir_escape(tmp_path: Path) -> None:
    index = tmp_path / "experiments" / "index.csv"
    index.parent.mkdir(parents=True)
    write_index(index, [_blank_row("escape", "../outside")])
    with pytest.raises(ValueError, match="escapes repository"):
        rebuild_index(tmp_path, index)


def test_rebuild_rejects_new_run_without_environment_identity(tmp_path: Path) -> None:
    index = tmp_path / "experiments" / "index.csv"
    index.parent.mkdir(parents=True)
    write_index(index, [])
    legacy = tmp_path / "artifacts" / "experiments" / "new-legacy-run"
    legacy.mkdir(parents=True)
    (legacy / "metrics.json").write_text(json.dumps({"best_top1": 0.5}), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks a complete environment identity"):
        rebuild_index(tmp_path, index)

    rows, audit = rebuild_index(tmp_path, index, allow_legacy_discovery=True)
    assert rows[0]["environment_hash"] == "unknown_legacy"
    assert audit.added_rows == 1


def test_read_index_rejects_duplicate_run_ids(tmp_path: Path) -> None:
    index = tmp_path / "index.csv"
    row = _blank_row("duplicate", "artifacts/experiments/duplicate")
    write_index(index, [row, row])
    with pytest.raises(ValueError, match="duplicate run_id"):
        read_index(index)


def test_read_index_text_preserves_multiline_notes(tmp_path: Path) -> None:
    index = tmp_path / "index.csv"
    row = _blank_row("multiline", "artifacts/experiments/multiline")
    row["notes"] = "line one\nline two"
    write_index(index, [row])
    parsed = read_index_text(index.read_text(encoding="utf-8"))
    assert parsed[0]["notes"] == "line one\nline two"
