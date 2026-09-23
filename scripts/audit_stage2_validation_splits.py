"""Audit two Stage 2 train-only validation folds without reading test images.

The second fold is a new experiment: an old PEFT checkpoint is not valid for
its validation set unless that checkpoint was trained strictly on that fold's
training partition.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.manifest import read_manifest, sha256_file  # noqa: E402
from aictrack1.data.splits import leakage_group  # noqa: E402


def _read_assignment(path: Path, sample_ids: set[str]) -> dict[str, str]:
    assignment: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["sample_id", "split"]:
            raise ValueError(f"unexpected split schema: {path}")
        for row in reader:
            sample_id, partition = row["sample_id"], row["split"]
            if sample_id in assignment:
                raise ValueError(f"duplicate split key in {path}: {sample_id}")
            if partition not in {"train", "validation", "excluded_bad_file"}:
                raise ValueError(f"unknown partition in {path}: {partition}")
            assignment[sample_id] = partition
    if set(assignment) != sample_ids:
        raise ValueError(f"split keys do not match train manifest: {path}")
    return assignment


def audit(manifest_path: Path, first_path: Path, second_path: Path) -> dict:
    records = read_manifest(manifest_path)
    if not records or any(record.source_split != "train" for record in records):
        raise ValueError("audit requires only labeled train-manifest records")
    sample_ids = {record.sample_id for record in records}
    if len(sample_ids) != len(records):
        raise ValueError("duplicate sample_id in train manifest")
    labels = {record.official_class_id for record in records}
    if not labels or "" in labels:
        raise ValueError("train manifest has missing labels")

    assignments = {
        "fold1": _read_assignment(first_path, sample_ids),
        "fold2": _read_assignment(second_path, sample_ids),
    }
    group_members: dict[str, list[str]] = defaultdict(list)
    hash_labels: dict[str, set[str]] = defaultdict(set)
    hash_members: dict[str, list[str]] = defaultdict(list)
    for record in records:
        group_members[leakage_group(record)].append(record.sample_id)
        hash_labels[record.sha256].add(record.official_class_id)
        hash_members[record.sha256].append(record.sample_id)
    conflict_ids = {
        sample_id
        for content_hash, distinct_labels in hash_labels.items()
        if len(distinct_labels) > 1
        for sample_id in hash_members[content_hash]
    }
    duplicate_group_ids = {
        sample_id for members in group_members.values() if len(members) > 1 for sample_id in members
    }

    result = {
        "scope": "Stage 2 labeled train manifest only; test not read",
        "manifest_sha256": sha256_file(manifest_path),
        "sample_count": len(records),
        "class_count": len(labels),
        "exact_conflict_sample_count": len(conflict_ids),
        "non_singleton_group_sample_count": len(duplicate_group_ids),
        "folds": {},
    }
    validation_ids = {}
    for name, path in (("fold1", first_path), ("fold2", second_path)):
        assignment = assignments[name]
        split_groups = [
            group_id
            for group_id, members in group_members.items()
            if len({assignment[sample_id] for sample_id in members}) != 1
        ]
        if split_groups:
            raise ValueError(f"{name} leaks {len(split_groups)} duplicate groups")
        totals = Counter(assignment.values())
        by_class: dict[str, Counter[str]] = defaultdict(Counter)
        for record in records:
            by_class[record.official_class_id][assignment[record.sample_id]] += 1
        validation_ids[name] = {
            sample_id for sample_id, partition in assignment.items() if partition == "validation"
        }
        result["folds"][name] = {
            "split_sha256": sha256_file(path),
            "counts": dict(totals),
            "duplicate_group_leakage_count": len(split_groups),
            "train_support_min": min(counts["train"] for counts in by_class.values()),
            "train_support_median": median(counts["train"] for counts in by_class.values()),
            "validation_support_min": min(counts["validation"] for counts in by_class.values()),
            "validation_support_median": median(counts["validation"] for counts in by_class.values()),
            "classes_without_train": sum(counts["train"] == 0 for counts in by_class.values()),
            "classes_without_validation": sum(counts["validation"] == 0 for counts in by_class.values()),
            "validation_exact_conflict_samples": len(validation_ids[name] & conflict_ids),
            "validation_exact_conflict_rate": len(validation_ids[name] & conflict_ids) / len(validation_ids[name]),
            "validation_non_singleton_group_samples": len(validation_ids[name] & duplicate_group_ids),
            "validation_non_singleton_group_rate": len(validation_ids[name] & duplicate_group_ids) / len(validation_ids[name]),
        }
    old_train = {sample_id for sample_id, partition in assignments["fold1"].items() if partition == "train"}
    result["cross_fold"] = {
        "validation_overlap": len(validation_ids["fold1"] & validation_ids["fold2"]),
        "validation_overlap_rate_of_fold2": len(validation_ids["fold1"] & validation_ids["fold2"]) / len(validation_ids["fold2"]),
        "fold2_validation_seen_by_fold1_training": len(validation_ids["fold2"] & old_train),
        "exact_conflict_validation_overlap": len(validation_ids["fold1"] & validation_ids["fold2"] & conflict_ids),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--fold1", required=True, type=Path)
    parser.add_argument("--fold2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.manifest, args.fold1, args.fold2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
