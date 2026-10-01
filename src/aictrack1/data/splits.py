"""Deterministic class-aware split with duplicate-cluster isolation."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from .manifest import SampleRecord, sha256_file


def leakage_group(record: SampleRecord) -> str:
    return record.near_duplicate_group or record.duplicate_group or f"sample-{record.sample_id}"


def build_grouped_stratified_split(
    records: list[SampleRecord],
    *,
    validation_fraction: float,
    seed: int,
    min_train_per_class: int,
    allowed_decode_statuses: frozenset[str] = frozenset({"ok"}),
    group_order: Literal["largest_first", "seeded_random"] = "largest_first",
    forbidden_validation_groups: frozenset[str] = frozenset(),
) -> dict[str, str]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if group_order not in {"largest_first", "seeded_random"}:
        raise ValueError(f"unknown group_order: {group_order}")
    labeled = [record for record in records if record.official_class_id]
    if len(labeled) != len(records):
        raise ValueError("only fully labeled training records can be split")
    eligible = [record for record in records if record.decode_status in allowed_decode_statuses]
    excluded = [record for record in records if record.decode_status not in allowed_decode_statuses]
    if not eligible:
        raise ValueError("no decodable records are eligible for splitting")
    totals = Counter(record.official_class_id for record in eligible)
    targets = {
        label: min(round(count * validation_fraction), max(0, count - min_train_per_class))
        for label, count in totals.items()
    }
    groups: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in eligible:
        groups[leakage_group(record)].append(record)

    def order_key(group_id: str) -> tuple[int, str]:
        digest = hashlib.sha256(f"{seed}\0{group_id}".encode("utf-8")).hexdigest()
        priority = -len(groups[group_id]) if group_order == "largest_first" else 0
        return (priority, digest)

    val_counts: Counter[str] = Counter()
    assignment: dict[str, str] = {}
    assignment.update({record.sample_id: "excluded_bad_file" for record in excluded})
    for group_id in sorted(groups, key=order_key):
        group_counts = Counter(record.official_class_id for record in groups[group_id])
        can_add = all(
            val_counts[label] + count <= targets[label]
            and totals[label] - (val_counts[label] + count) >= min_train_per_class
            for label, count in group_counts.items()
        )
        improves = any(val_counts[label] < targets[label] for label in group_counts)
        split = "validation" if can_add and improves and group_id not in forbidden_validation_groups else "train"
        if split == "validation":
            val_counts.update(group_counts)
        for record in groups[group_id]:
            assignment[record.sample_id] = split

    if len(assignment) != len(records):
        raise RuntimeError("split assignment lost samples")
    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in eligible:
        group_splits[leakage_group(record)].add(assignment[record.sample_id])
    if any(len(values) != 1 for values in group_splits.values()):
        raise RuntimeError("duplicate group leaked across train and validation")
    return assignment


def write_split(assignment: dict[str, str], path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "split"], lineterminator="\n")
        writer.writeheader()
        for sample_id in sorted(assignment):
            writer.writerow({"sample_id": sample_id, "split": assignment[sample_id]})
    return sha256_file(output)
