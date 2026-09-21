from __future__ import annotations

import pytest

from aictrack1.data.manifest import SampleRecord
from aictrack1.data.overlap import exact_cross_source_overlap


def _record(source: str, sample_id: str, path: str, digest: str) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        relative_path=path,
        source_split=source,
        sha256=digest,
        decode_status="ok",
    )


def test_exact_cross_source_overlap_reports_shared_hashes() -> None:
    train = [
        _record("train", "tr-1", "0000/a.jpg", "a" * 64),
        _record("train", "tr-2", "0001/b.jpg", "b" * 64),
    ]
    test = [
        _record("test", "te-1", "x.jpg", "b" * 64),
        _record("test", "te-2", "y.jpg", "c" * 64),
    ]

    summary, rows = exact_cross_source_overlap(train, test)

    assert summary["overlap_hash_count"] == 1
    assert summary["overlapping_train_sample_count"] == 1
    assert summary["overlapping_test_sample_count"] == 1
    assert rows == [
        {
            "sha256": "b" * 64,
            "train_count": 1,
            "test_count": 1,
            "train_paths": "0001/b.jpg",
            "test_paths": "x.jpg",
        }
    ]


def test_exact_cross_source_overlap_rejects_wrong_source_manifest() -> None:
    with pytest.raises(ValueError, match="non-train"):
        exact_cross_source_overlap(
            [_record("validation", "tr", "a.jpg", "a" * 64)],
            [_record("test", "te", "b.jpg", "b" * 64)],
        )


def test_exact_cross_source_overlap_requires_complete_hashes() -> None:
    with pytest.raises(ValueError, match="complete SHA-256"):
        exact_cross_source_overlap(
            [_record("train", "tr", "a.jpg", "")],
            [_record("test", "te", "b.jpg", "b" * 64)],
        )
