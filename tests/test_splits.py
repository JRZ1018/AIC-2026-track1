from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from aictrack1.data.audit import audit_records
from aictrack1.data.contracts import load_dataset_config
from aictrack1.data.manifest import build_manifest
from aictrack1.data.splits import build_grouped_stratified_split, leakage_group


def test_split_is_deterministic_and_has_no_duplicate_leakage(synthetic_dataset: tuple[Path, Path]) -> None:
    _, config_path = synthetic_dataset
    config = load_dataset_config(config_path)
    records, _, _ = audit_records(build_manifest(config, "train"), near_hamming=3, max_bucket_size=100)
    kwargs = dict(validation_fraction=0.25, seed=7, min_train_per_class=2)
    first = build_grouped_stratified_split(records, **kwargs)
    second = build_grouped_stratified_split(list(reversed(records)), **kwargs)
    assert first == second
    groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        groups[leakage_group(record)].add(first[record.sample_id])
    assert all(len(splits) == 1 for splits in groups.values())
    assert "validation" in set(first.values())
    assert list(first.values()).count("excluded_bad_file") == 2
    train_counts = Counter(record.official_class_id for record in records if first[record.sample_id] == "train")
    assert all(count >= 2 for count in train_counts.values())


def test_seeded_random_group_order_is_deterministic_and_isolated(
    synthetic_dataset: tuple[Path, Path],
) -> None:
    _, config_path = synthetic_dataset
    config = load_dataset_config(config_path)
    records, _, _ = audit_records(build_manifest(config, "train"), near_hamming=3, max_bucket_size=100)
    kwargs = dict(validation_fraction=0.25, seed=20260924, min_train_per_class=2, group_order="seeded_random")
    first = build_grouped_stratified_split(records, **kwargs)
    assert first == build_grouped_stratified_split(list(reversed(records)), **kwargs)
    groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        groups[leakage_group(record)].add(first[record.sample_id])
    assert all(len(partitions) == 1 for partitions in groups.values())
    assert "validation" in set(first.values())


def test_prior_validation_groups_can_be_excluded_from_new_fold(
    synthetic_dataset: tuple[Path, Path],
) -> None:
    _, config_path = synthetic_dataset
    config = load_dataset_config(config_path)
    records, _, _ = audit_records(build_manifest(config, "train"), near_hamming=3, max_bucket_size=100)
    kwargs = dict(validation_fraction=0.25, seed=7, min_train_per_class=2, group_order="seeded_random")
    prior = build_grouped_stratified_split(records, **kwargs)
    prior_validation_groups = frozenset(
        leakage_group(record) for record in records if prior[record.sample_id] == "validation"
    )
    newer = build_grouped_stratified_split(
        records,
        **{**kwargs, "seed": 20260925},
        forbidden_validation_groups=prior_validation_groups,
    )
    assert all(
        newer[record.sample_id] != "validation"
        for record in records
        if leakage_group(record) in prior_validation_groups
    )
    assert newer == build_grouped_stratified_split(
        list(reversed(records)),
        **{**kwargs, "seed": 20260925},
        forbidden_validation_groups=prior_validation_groups,
    )
