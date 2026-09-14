from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from aictrack1.data.long_tail_split import (
    LONG_TAIL_EVAL,
    LONG_TAIL_TRAIN,
    LongTailSplitConfig,
    build_long_tail_split,
    read_split_assignments,
    split_duplicate_cluster_violations,
    summarize_partition,
    write_long_tail_split,
)
from aictrack1.data.manifest import SampleRecord, read_manifest
from aictrack1.evaluation.long_tail import HEAD, MID, TAIL, stratify_classes


def _scratch_dir() -> Path:
    """Create a workspace-local scratch directory for file round trips."""
    path = Path("artifacts/pytest/scratch") / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def _record(index: int, class_index: int, *, near_group: str = "") -> SampleRecord:
    return SampleRecord(
        sample_id=f"{index:08d}",
        relative_path=f"{class_index:04d}/{index:08d}.jpg",
        source_split="train",
        observed_label=f"{class_index:04d}",
        internal_class_index=class_index,
        official_class_id=f"{class_index:04d}",
        decode_status="ok",
        near_duplicate_group=near_group,
    )


def _make_dataset(counts: list[int], *, seed_offset: int = 0) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    counter = seed_offset
    for class_index, count in enumerate(counts):
        for _ in range(count):
            records.append(_record(counter, class_index))
            counter += 1
    return records


def _config(**overrides) -> LongTailSplitConfig:
    payload = {
        "reference_partition": "train",
        "train_fraction": 0.2,
        "eval_fraction": 0.15,
        "tail_retention": 0.2,
        "mid_retention": 0.4,
        "head_fraction": 0.34,
        "mid_fraction": 0.33,
        "min_samples_per_class": 1,
        "seed": 11,
    }
    payload.update(overrides)
    return LongTailSplitConfig(**payload)


def test_build_long_tail_split_is_deterministic_and_covers_every_class() -> None:
    counts = [40, 30, 20, 10, 5, 3]
    records = _make_dataset(counts)
    assignments = {record.sample_id: "train" for record in records}
    strata = stratify_classes(counts, head_fraction=0.34, mid_fraction=0.33, seed=11)

    first = build_long_tail_split(
        records,
        assignments,
        config=_config(),
        stratification_assignments=strata.assignments,
        class_counts=tuple(counts),
    )
    second = build_long_tail_split(
        records,
        assignments,
        config=_config(),
        stratification_assignments=strata.assignments,
        class_counts=tuple(counts),
    )
    assert first == second
    train_counts = summarize_partition(
        records, first, partition=LONG_TAIL_TRAIN, num_classes=len(counts)
    )
    eval_counts = summarize_partition(
        records, first, partition=LONG_TAIL_EVAL, num_classes=len(counts)
    )
    # Every class keeps training samples, and the mid plus at least one tail class
    # feed the stratified diagnostics (head classes are the largest and are covered
    # by the balanced reference pass; the smallest class can legitimately round down
    # to zero evaluation samples and is reported as unevaluated).
    assert all(value >= 1 for value in train_counts), train_counts
    assert sum(eval_counts[2:]) >= 2, eval_counts
    # The long-tail partitions keep a strict subset: the rest of the frozen training
    # partition stays out of the stress test.
    assert 0 < sum(train_counts) + sum(eval_counts) < len(records)
    assert set(first.values()) == {LONG_TAIL_TRAIN, LONG_TAIL_EVAL}


def test_build_long_tail_split_respects_retention_schedule() -> None:
    counts = [100, 100, 100, 100]
    records = _make_dataset(counts)
    assignments = {record.sample_id: "train" for record in records}
    strata = stratify_classes(counts, head_fraction=0.25, mid_fraction=0.25, seed=3)
    assert strata.assignments == (HEAD, MID, TAIL, TAIL)
    assignment = build_long_tail_split(
        records,
        assignments,
        config=_config(head_fraction=0.25, mid_fraction=0.25, tail_retention=0.1, mid_retention=0.5),
        stratification_assignments=strata.assignments,
        class_counts=tuple(counts),
    )
    train_counts = summarize_partition(records, assignment, partition=LONG_TAIL_TRAIN, num_classes=4)
    eval_counts = summarize_partition(records, assignment, partition=LONG_TAIL_EVAL, num_classes=4)
    # head keeps everything, mid keeps half, and both tail classes keep a tenth;
    # the remainder feeds the evaluation partition, which only needs to cover the
    # stratified diagnostics (head classes are the largest and are evaluated in
    # isolation by the balanced reference).
    assert train_counts[0] == 100
    assert train_counts[1] == 50
    assert train_counts[2] == 10
    assert train_counts[3] == 10
    assert eval_counts[1] + eval_counts[2] + eval_counts[3] > 0


def test_build_long_tail_split_keeps_duplicate_clusters_together() -> None:
    counts = [20, 20, 20, 20]
    records = _make_dataset(counts)
    # One near-duplicate cluster spanning three classes must never be split.
    for record in records[:3]:
        records[records.index(record)] = SampleRecord(
            **{
                **record.__dict__,
                "near_duplicate_group": "near-shared",
            }
        )
    assignments = {record.sample_id: "train" for record in records}
    strata = stratify_classes(counts, head_fraction=0.25, mid_fraction=0.25, seed=5)
    assignment = build_long_tail_split(
        records,
        assignments,
        config=_config(head_fraction=0.25, mid_fraction=0.25),
        stratification_assignments=strata.assignments,
        class_counts=tuple(counts),
    )
    assert split_duplicate_cluster_violations(records, assignment) == 0
    partitions = {assignment[record.sample_id] for record in records if record.near_duplicate_group}
    assert len(partitions) == 1


def test_build_long_tail_split_honours_minimum_floor() -> None:
    counts = [100, 80, 6, 4]
    records = _make_dataset(counts)
    assignments = {record.sample_id: "train" for record in records}
    strata = stratify_classes(counts, head_fraction=0.25, mid_fraction=0.25, seed=9)
    assert strata.assignments == (HEAD, MID, TAIL, TAIL)
    # 5% of a six-sample class rounds to zero, so the floor is what keeps it alive.
    assignment = build_long_tail_split(
        records,
        assignments,
        config=_config(
            head_fraction=0.25,
            mid_fraction=0.25,
            tail_retention=0.05,
            min_samples_per_class=2,
        ),
        stratification_assignments=strata.assignments,
        class_counts=tuple(counts),
    )
    train_counts = summarize_partition(records, assignment, partition=LONG_TAIL_TRAIN, num_classes=4)
    assert train_counts[2] >= 2
    assert train_counts[3] >= 2


def test_build_long_tail_split_rejects_missing_partition_and_empty_result() -> None:
    records = _make_dataset([20, 15, 10, 5])
    strata = stratify_classes([20, 15, 10, 5], head_fraction=0.25, mid_fraction=0.25, seed=1)
    with pytest.raises(ValueError, match="no record belongs"):
        build_long_tail_split(
            records,
            {record.sample_id: "validation" for record in records},
            config=_config(),
            stratification_assignments=strata.assignments,
            class_counts=(20, 15, 10, 5),
        )
    with pytest.raises(ValueError, match="min_samples_per_class"):
        build_long_tail_split(
            records,
            {record.sample_id: "train" for record in records},
            config=_config(min_samples_per_class=0),
            stratification_assignments=strata.assignments,
            class_counts=(20, 15, 10, 5),
        )


def test_write_and_read_long_tail_split_round_trip() -> None:
    assignment = {"b": LONG_TAIL_EVAL, "a": LONG_TAIL_TRAIN}
    path = _scratch_dir() / "long_tail_split.csv"
    digest = write_long_tail_split(assignment, path)
    assert len(digest) == 64
    assert path.read_text(encoding="utf-8").splitlines() == [
        "sample_id,long_tail_split",
        "a,long_tail_train",
        "b,long_tail_eval",
    ]


def test_read_split_assignments_rejects_duplicates() -> None:
    path = _scratch_dir() / "split.csv"
    path.write_text("sample_id,split\na,train\na,validation\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate sample_id"):
        read_split_assignments(path)


def test_read_manifest_survives_round_trip() -> None:
    from aictrack1.data.manifest import write_manifest

    records = _make_dataset([2, 1])
    path = _scratch_dir() / "samples.csv"
    write_manifest(records, path)
    loaded = read_manifest(path)
    assert [record.sample_id for record in loaded] == [record.sample_id for record in records]
    assert [record.internal_class_index for record in loaded] == [0, 0, 1]
