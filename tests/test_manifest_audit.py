from __future__ import annotations

from pathlib import Path

from aictrack1.data.audit import audit_records, write_audit_report
from aictrack1.data.contracts import load_dataset_config
from aictrack1.data.manifest import SampleRecord, build_manifest, read_manifest, sha256_file, write_manifest


def _source_hashes(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in root.rglob("*") if path.is_file()}


def test_manifest_and_audit_are_deterministic_and_read_only(synthetic_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    root, config_path = synthetic_dataset
    before = _source_hashes(root)
    config = load_dataset_config(config_path)
    records = build_manifest(config, "train")
    assert any(record.decode_status == "decode_error" for record in records)
    assert any(record.decode_status == "truncated" for record in records)
    manifest_path = tmp_path / "plain_manifest.csv"
    first_hash = write_manifest(records, manifest_path)
    assert first_hash == write_manifest(records, manifest_path)
    assert read_manifest(manifest_path) == records
    summary = write_audit_report(config, "train")
    assert summary["bad_file_count"] == 2
    assert summary["exact_duplicate_groups"] >= 1
    assert summary["near_duplicate_groups"] >= 1
    assert before == _source_hashes(root)


def test_same_content_conflicting_labels_are_reported() -> None:
    records = [
        SampleRecord("a", "a.png", "train", "0000", 0, "0000", sha256="same", dhash="0" * 16, decode_status="ok"),
        SampleRecord("b", "b.png", "train", "0001", 1, "0001", sha256="same", dhash="0" * 16, decode_status="ok"),
    ]
    _, summary, conflicts = audit_records(records)
    assert summary["same_content_label_conflicts"] == 1
    assert conflicts[0]["labels"] == "0000;0001"
