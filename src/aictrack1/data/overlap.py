"""Integrity-only checks for exact content overlap between data sources."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from .manifest import SampleRecord


def exact_cross_source_overlap(
    train_records: Sequence[SampleRecord],
    test_records: Sequence[SampleRecord],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Report exact SHA-256 overlap without using test labels or predictions."""

    if any(record.source_split != "train" for record in train_records):
        raise ValueError("train manifest contains a non-train record")
    if any(record.source_split != "test" for record in test_records):
        raise ValueError("test manifest contains a non-test record")
    if any(not record.sha256 for record in (*train_records, *test_records)):
        raise ValueError("cross-source overlap requires complete SHA-256 values")

    train_by_hash: dict[str, list[SampleRecord]] = defaultdict(list)
    test_by_hash: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in train_records:
        train_by_hash[record.sha256].append(record)
    for record in test_records:
        test_by_hash[record.sha256].append(record)

    rows: list[dict[str, Any]] = []
    for digest in sorted(set(train_by_hash) & set(test_by_hash)):
        train_group = train_by_hash[digest]
        test_group = test_by_hash[digest]
        rows.append(
            {
                "sha256": digest,
                "train_count": len(train_group),
                "test_count": len(test_group),
                "train_paths": ";".join(sorted(record.relative_path for record in train_group)),
                "test_paths": ";".join(sorted(record.relative_path for record in test_group)),
            }
        )

    summary = {
        "schema": "aictrack1.cross_source_exact_overlap.v1",
        "train_sample_count": len(train_records),
        "test_sample_count": len(test_records),
        "overlap_hash_count": len(rows),
        "overlapping_train_sample_count": sum(int(row["train_count"]) for row in rows),
        "overlapping_test_sample_count": sum(int(row["test_count"]) for row in rows),
        "scope": "integrity-only SHA-256 comparison; no test labels, predictions, or statistics used",
    }
    return summary, rows
