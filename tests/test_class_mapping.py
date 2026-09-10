from __future__ import annotations

from pathlib import Path

import pytest

from aictrack1.data.class_mapping import ClassEntry, ClassMapping


def test_leading_zero_round_trip(synthetic_dataset: tuple[Path, Path]) -> None:
    root, _ = synthetic_dataset
    mapping = ClassMapping.from_csv(root / "metadata" / "classes.csv")
    assert mapping.to_internal("0003") == 3
    assert mapping.to_official(3) == "0003"
    assert len(mapping.digest()) == 64


def test_mapping_rejects_duplicate_and_noncontiguous_indices() -> None:
    with pytest.raises(ValueError, match="duplicate official"):
        ClassMapping([ClassEntry("0001", 0), ClassEntry("0001", 1)])
    with pytest.raises(ValueError, match="contiguous"):
        ClassMapping([ClassEntry("0001", 0), ClassEntry("0002", 2)])


def test_directory_mapping_is_sorted_lossless_and_writable(tmp_path: Path) -> None:
    train = tmp_path / "train"
    for label in ("0010", "0002", "0000"):
        (train / label).mkdir(parents=True)
    mapping = ClassMapping.from_train_directories(train)
    assert [entry.official_class_id for entry in mapping.entries] == ["0000", "0002", "0010"]
    output = tmp_path / "derived" / "classes.csv"
    mapping.write_csv(output)
    restored = ClassMapping.from_csv(output)
    assert restored.entries == mapping.entries


def test_directory_mapping_rejects_ambiguous_names(tmp_path: Path) -> None:
    (tmp_path / "train" / "class-one").mkdir(parents=True)
    with pytest.raises(ValueError, match="four digits"):
        ClassMapping.from_train_directories(tmp_path / "train")
