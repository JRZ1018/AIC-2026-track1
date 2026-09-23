from __future__ import annotations

import csv
from pathlib import Path

import pytest

from aictrack1.data.manifest import SampleRecord, write_manifest
from scripts.audit_stage2_validation_splits import audit


def _write_split(path: Path, partitions: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "split"])
        writer.writerows(sorted(partitions.items()))


def test_audit_counts_conflicts_and_detects_cross_fold_reuse(tmp_path: Path) -> None:
    records = [
        SampleRecord("a", "0000/a.jpg", "train", official_class_id="0000", sha256="same", near_duplicate_group="g"),
        SampleRecord("b", "0001/b.jpg", "train", official_class_id="0001", sha256="same", near_duplicate_group="g"),
        SampleRecord("c", "0000/c.jpg", "train", official_class_id="0000", sha256="other"),
        SampleRecord("d", "0001/d.jpg", "train", official_class_id="0001", sha256="last"),
    ]
    manifest = tmp_path / "samples.csv"
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    write_manifest(records, manifest)
    _write_split(first, {"a": "validation", "b": "validation", "c": "train", "d": "train"})
    _write_split(second, {"a": "train", "b": "train", "c": "validation", "d": "validation"})
    result = audit(manifest, first, second)
    assert result["exact_conflict_sample_count"] == 2
    assert result["folds"]["fold1"]["validation_exact_conflict_samples"] == 2
    assert result["cross_fold"]["fold2_validation_seen_by_fold1_training"] == 2


def test_audit_rejects_duplicate_group_leakage(tmp_path: Path) -> None:
    records = [
        SampleRecord("a", "0000/a.jpg", "train", official_class_id="0000", sha256="a", near_duplicate_group="g"),
        SampleRecord("b", "0001/b.jpg", "train", official_class_id="0001", sha256="b", near_duplicate_group="g"),
    ]
    manifest = tmp_path / "samples.csv"
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    write_manifest(records, manifest)
    _write_split(first, {"a": "train", "b": "validation"})
    _write_split(second, {"a": "train", "b": "train"})
    with pytest.raises(ValueError, match="leaks"):
        audit(manifest, first, second)
