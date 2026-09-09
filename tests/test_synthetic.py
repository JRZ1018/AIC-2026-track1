from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from aictrack1.data.manifest import sha256_file
from aictrack1.data.synthetic import (
    SyntheticConfig,
    class_counts,
    generate_synthetic_dataset,
    noise_probability,
    sample_alternate_label,
)


def _file_hashes(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file()}


def test_generation_is_byte_deterministic(tmp_path: Path) -> None:
    common = dict(seed=99, num_classes=4, classes_per_family=2, max_samples_per_class=5, image_size=32, noise_mode="symmetric", noise_rate=0.5, exact_duplicate_count=1, near_duplicate_count=1, corrupt_count=1, truncated_count=1)
    left, right = tmp_path / "left", tmp_path / "right"
    generate_synthetic_dataset(SyntheticConfig(output_root=left, **common))
    generate_synthetic_dataset(SyntheticConfig(output_root=right, **common))
    assert _file_hashes(left) == _file_hashes(right)


def test_long_tail_and_noise_truth(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    config = SyntheticConfig(output_root=root, seed=4, num_classes=8, classes_per_family=2, max_samples_per_class=20, imbalance_ratio=5, noise_mode="symmetric", noise_rate=1.0, exact_duplicate_count=0, near_duplicate_count=0, corrupt_count=0, truncated_count=0)
    summary = generate_synthetic_dataset(config)
    counts = class_counts(config)
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] >= 1
    assert summary["realized_noisy_count"] == summary["base_sample_count"]
    with (root / "metadata" / "samples.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["observed_label"] != row["clean_label"] for row in rows)


@pytest.mark.parametrize("noise_mode", ["symmetric", "pairflip", "family", "instance", "class_dependent"])
def test_all_noise_modes_produce_valid_labels(tmp_path: Path, noise_mode: str) -> None:
    root = tmp_path / noise_mode
    config = SyntheticConfig(output_root=root, seed=17, num_classes=6, classes_per_family=3, max_samples_per_class=12, noise_mode=noise_mode, noise_rate=0.7, exact_duplicate_count=0, near_duplicate_count=0, corrupt_count=0, truncated_count=0)
    generate_synthetic_dataset(config)
    with (root / "metadata" / "samples.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["is_noisy"] == "1" for row in rows)
    assert all(0 <= int(row["observed_label"]) < config.num_classes for row in rows)
    assert all((row["observed_label"] != row["clean_label"]) == (row["is_noisy"] == "1") for row in rows)


def test_noise_mode_semantics() -> None:
    rng = np.random.default_rng(5)
    base = dict(output_root=Path("unused"), num_classes=6, classes_per_family=3)
    pair = SyntheticConfig(**base, noise_mode="pairflip")
    assert sample_alternate_label(pair, 2, 0.5, rng) == 3
    family = SyntheticConfig(**base, noise_mode="family")
    assert sample_alternate_label(family, 1, 0.5, rng) in {0, 2}
    assert sample_alternate_label(family, 4, 0.5, rng) in {3, 5}
    symmetric = SyntheticConfig(**base, noise_mode="symmetric")
    assert all(sample_alternate_label(symmetric, 3, 0.5, rng) != 3 for _ in range(20))

    instance = SyntheticConfig(**base, noise_mode="instance", noise_rate=0.5)
    assert noise_probability(instance, 0, 0.9) > noise_probability(instance, 0, 0.1)
    class_dependent = SyntheticConfig(**base, noise_mode="class_dependent", noise_rate=0.5)
    assert noise_probability(class_dependent, 5, 0.5) > noise_probability(class_dependent, 0, 0.5)
