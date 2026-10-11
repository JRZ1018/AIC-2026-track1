"""Deterministic loss controls for exact cross-label image conflicts."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import torch


def conflict_group_weight_lookup(
    manifest_path: str | Path,
    conflicts_path: str | Path,
    split_path: str | Path,
    *,
    partition: str | None,
    minimum_size: int = 0,
) -> tuple[torch.Tensor, dict[str, int | str]]:
    """Return a manifest-indexed lookup for X1 group-normalized loss.

    Only hashes explicitly listed by the current-stage conflict audit are used.  A
    group size is counted within the selected partition (normally ``train``), so a
    duplicate that is held out for validation does not reduce the training weight of
    its remaining member.  Non-conflict samples and groups with one selected member
    receive weight one.  The lookup is intentionally independent of labels and test
    data.
    """

    manifest = Path(manifest_path)
    conflicts = Path(conflicts_path)
    split = Path(split_path)
    conflict_hashes: set[str] = set()
    with conflicts.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "sha256" not in reader.fieldnames:
            raise ValueError("conflict audit must contain a sha256 column")
        conflict_hashes = {str(row["sha256"]) for row in reader}
    assignments: dict[str, str] = {}
    with split.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"sample_id", "split"}.issubset(reader.fieldnames):
            raise ValueError("split must contain sample_id and split columns")
        assignments = {str(row["sample_id"]): str(row["split"]) for row in reader}

    rows: list[tuple[int, str, str]] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "sha256"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("manifest must contain sample_id and sha256 columns")
        for index, row in enumerate(reader):
            sample_id = str(row["sample_id"])
            sha256 = str(row["sha256"])
            if sha256 not in conflict_hashes:
                continue
            assigned = assignments.get(sample_id)
            if partition is not None and assigned != partition:
                continue
            rows.append((index, sample_id, sha256))

    group_counts = Counter(sha256 for _, _, sha256 in rows)
    if minimum_size < 0:
        raise ValueError("minimum_size must be non-negative")
    lookup = torch.ones(
        (max(minimum_size, max((index for index, _, _ in rows), default=-1) + 1),),
        dtype=torch.float32,
    )
    for index, _, sha256 in rows:
        count = group_counts[sha256]
        if count > 1:
            lookup[index] = 1.0 / float(count)
    diagnostics: dict[str, int | str] = {
        "mode": "group_normalized",
        "partition": partition or "all",
        "conflict_group_count": len(group_counts),
        "conflict_sample_count": len(rows),
        "multi_member_group_count": sum(1 for count in group_counts.values() if count > 1),
        "lookup_size": int(lookup.numel()),
    }
    return lookup, diagnostics
