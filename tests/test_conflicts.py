from __future__ import annotations

import csv

import pytest
import torch

from aictrack1.methods.conflicts import conflict_group_weight_lookup


def _write_csv(path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_conflict_group_lookup_normalizes_only_selected_partition(tmp_path) -> None:
    manifest = tmp_path / "samples.csv"
    conflicts = tmp_path / "conflicts.csv"
    split = tmp_path / "split.csv"
    _write_csv(
        manifest,
        ["sample_id", "sha256"],
        [
            {"sample_id": "a", "sha256": "h1"},
            {"sample_id": "b", "sha256": "h1"},
            {"sample_id": "c", "sha256": "h2"},
            {"sample_id": "d", "sha256": "h3"},
        ],
    )
    _write_csv(
        conflicts,
        ["sha256"],
        [{"sha256": "h1"}, {"sha256": "h2"}],
    )
    _write_csv(
        split,
        ["sample_id", "split"],
        [
            {"sample_id": "a", "split": "train"},
            {"sample_id": "b", "split": "validation"},
            {"sample_id": "c", "split": "train"},
            {"sample_id": "d", "split": "train"},
        ],
    )
    lookup, diagnostics = conflict_group_weight_lookup(
        manifest, conflicts, split, partition="train", minimum_size=4
    )
    assert torch.equal(lookup, torch.ones(4))
    assert diagnostics["conflict_group_count"] == 2
    assert diagnostics["multi_member_group_count"] == 0

    lookup_all, diagnostics_all = conflict_group_weight_lookup(
        manifest, conflicts, split, partition=None
    )
    assert lookup_all[0] == pytest.approx(0.5)
    assert lookup_all[1] == pytest.approx(0.5)
    assert lookup_all[2] == pytest.approx(1.0)
    assert diagnostics_all["multi_member_group_count"] == 1


def test_conflict_group_lookup_rejects_bad_audit_schema(tmp_path) -> None:
    manifest = tmp_path / "samples.csv"
    conflicts = tmp_path / "conflicts.csv"
    split = tmp_path / "split.csv"
    _write_csv(manifest, ["sample_id", "sha256"], [])
    _write_csv(conflicts, ["hash"], [])
    _write_csv(split, ["sample_id", "split"], [])
    with pytest.raises(ValueError, match="sha256"):
        conflict_group_weight_lookup(manifest, conflicts, split, partition="train")
