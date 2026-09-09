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
