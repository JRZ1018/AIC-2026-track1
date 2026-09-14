"""Deterministic long-tailed stress split built from the current-stage train part.

The split is a migration-engineering artefact. It never replaces the fixed
noisy-validation selection set and it never reads competition test data: every
sample it keeps comes from the training partition of the already frozen split
file, and duplicate-cluster isolation is preserved.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .manifest import SampleRecord, sha256_file
from .splits import leakage_group

LONG_TAIL_TRAIN = "long_tail_train"
LONG_TAIL_EVAL = "long_tail_eval"


@dataclass(frozen=True)
class LongTailSplitConfig:
    reference_partition: str
    train_fraction: float
    eval_fraction: float
    tail_retention: float
    mid_retention: float
    head_fraction: float
    mid_fraction: float
    min_samples_per_class: int
    seed: int


def read_split_assignments(path: str | Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"sample_id", "split"}.issubset(reader.fieldnames):
            raise ValueError("split file requires sample_id and split columns")
        for row in reader:
            sample_id = str(row["sample_id"])
            if sample_id in assignments:
                raise ValueError(f"duplicate sample_id in split file: {sample_id}")
            assignments[sample_id] = str(row["split"])
    return assignments


def _shuffle_key(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{sample_id}".encode("utf-8")).hexdigest()


def _select_class_positions(
    positions: list[int],
    *,
    train_quota: int,
    eval_quota: int,
    index_to_group: dict[int, str],
    key: dict[int, str],
) -> tuple[list[int], list[int]]:
    """Assign whole duplicate clusters to long-tail train or long-tail eval.

    The cluster is the smallest indivisible unit, matching the fixed split rule.
    A multi-member cluster can therefore overshoot one quota by a few samples;
    that is preferred over leaking a duplicate cluster across the two partitions.
    """
    ordered = sorted(positions, key=lambda index: (key[index], index))
    train: list[int] = []
    evaluation: list[int] = []
    train_groups: set[str] = set()
    eval_groups: set[str] = set()
    for index in ordered:
        group = index_to_group[index]
        if group in eval_groups:
            evaluation.append(index)
            continue
        if group in train_groups:
            train.append(index)
            continue
        train_room = len(train) < train_quota
        eval_room = len(evaluation) < eval_quota
        if not train_room and not eval_room:
            break
        go_to_train = train_room and (not eval_room or key[index][-1] < "8")
        if go_to_train:
            train.append(index)
            train_groups.add(group)
        else:
            evaluation.append(index)
            eval_groups.add(group)
    return train, evaluation


def build_long_tail_split(
    records: list[SampleRecord],
    assignments: dict[str, str],
    *,
    config: LongTailSplitConfig,
    stratification_assignments: tuple[str, ...],
    class_counts: tuple[int, ...],
) -> dict[str, str]:
    """Return ``sample_id -> {long_tail_train, long_tail_eval}`` for the kept part."""
    if len(stratification_assignments) != len(class_counts):
        raise ValueError("stratification and class counts disagree")
    if not 0 < config.train_fraction < 1 or not 0 < config.eval_fraction < 1:
        raise ValueError("train_fraction and eval_fraction must be strictly between 0 and 1")
    if not 0 < config.tail_retention <= 1 or not 0 < config.mid_retention <= 1:
        raise ValueError("retention rates must be in (0, 1]")
    if config.min_samples_per_class < 1:
        raise ValueError("min_samples_per_class must be at least one")
    num_classes = len(class_counts)
    reference = [
        record
        for record in records
        if assignments.get(record.sample_id) == config.reference_partition
    ]
    if not reference:
        raise ValueError(f"no record belongs to partition {config.reference_partition!r}")
    unlabeled = [
        record.sample_id for record in reference if record.internal_class_index is None
    ]
    if unlabeled:
        raise ValueError(f"{len(unlabeled)} selected records carry no class label")

    by_class: dict[int, list[int]] = {}
    index_to_group: dict[int, str] = {}
    key: dict[int, str] = {}
    for position, record in enumerate(reference):
        by_class.setdefault(int(record.internal_class_index), []).append(position)
        index_to_group[position] = leakage_group(record)
        key[position] = _shuffle_key(config.seed, record.sample_id)

    retention = {"head": 1.0, "mid": float(config.mid_retention), "tail": float(config.tail_retention)}
    assignment: dict[str, str] = {}
    for class_index in range(num_classes):
        positions = by_class.get(class_index, [])
        available = len(positions)
        rate = retention[stratification_assignments[class_index]]
        intended = max(config.min_samples_per_class, round(available * rate))
        train_quota = min(available, intended)
        eval_quota = min(available - train_quota, round(available * config.eval_fraction))
        if train_quota + eval_quota == 0:
            raise ValueError(
                f"class {class_index} loses every sample; raise min_samples_per_class"
            )
        chosen_train, chosen_eval = _select_class_positions(
            positions,
            train_quota=train_quota,
            eval_quota=eval_quota,
            index_to_group=index_to_group,
            key=key,
        )
        for position in chosen_train:
            assignment[reference[position].sample_id] = LONG_TAIL_TRAIN
        for position in chosen_eval:
            assignment[reference[position].sample_id] = LONG_TAIL_EVAL
    _reconcile_duplicate_clusters(reference, assignment, key=key)
    return assignment


def _reconcile_duplicate_clusters(
    reference: list[SampleRecord],
    assignment: dict[str, str],
    *,
    key: dict[int, str],
) -> None:
    """Force every duplicate cluster into exactly one long-tail partition.

    Per-class quota selection can split one near-duplicate cluster when its
    members carry conflicting labels, so a global pass moves a spanning cluster
    as a whole. The destination follows the cluster majority and falls back to a
    deterministic hash when the split is even.
    """
    clusters: dict[str, list[int]] = {}
    for position, record in enumerate(reference):
        if record.sample_id in assignment:
            clusters.setdefault(leakage_group(record), []).append(position)
    for group, positions in clusters.items():
        partitions = {assignment[reference[position].sample_id] for position in positions}
        if len(partitions) < 2:
            continue
        train_members = sum(
            1
            for position in positions
            if assignment[reference[position].sample_id] == LONG_TAIL_TRAIN
        )
        eval_members = len(positions) - train_members
        if train_members == eval_members:
            destination = LONG_TAIL_TRAIN if key[positions[0]][-1] < "8" else LONG_TAIL_EVAL
        else:
            destination = LONG_TAIL_TRAIN if train_members > eval_members else LONG_TAIL_EVAL
        for position in positions:
            assignment[reference[position].sample_id] = destination


def summarize_partition(
    records: list[SampleRecord],
    assignments: dict[str, str],
    *,
    partition: str,
    num_classes: int,
) -> tuple[int, ...]:
    """Return per-class sample counts for one long-tail partition."""
    counts = Counter(
        int(record.internal_class_index)
        for record in records
        if assignments.get(record.sample_id) == partition
    )
    if any(index < 0 or index >= num_classes for index in counts):
        raise ValueError("partition contains a class index outside num_classes")
    return tuple(counts.get(index, 0) for index in range(num_classes))


def split_duplicate_cluster_violations(
    records: list[SampleRecord],
    assignment: dict[str, str],
) -> int:
    """Count duplicate clusters that leaked across the two long-tail partitions."""
    clusters: dict[str, set[str]] = {}
    for record in records:
        partition = assignment.get(record.sample_id)
        if partition is None:
            continue
        clusters.setdefault(leakage_group(record), set()).add(partition)
    return sum(1 for partitions in clusters.values() if len(partitions) > 1)


def write_long_tail_split(assignment: dict[str, str], path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "long_tail_split"], lineterminator="\n"
        )
        writer.writeheader()
        for sample_id in sorted(assignment):
            writer.writerow({"sample_id": sample_id, "long_tail_split": assignment[sample_id]})
    return sha256_file(output)
