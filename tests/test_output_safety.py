from __future__ import annotations

from pathlib import Path

import csv
import pytest

from aictrack1.data.audit import write_audit_report
from aictrack1.data.contracts import ensure_output_path_is_safe, load_dataset_config


def test_all_outputs_are_rejected_inside_dataset(
    synthetic_dataset: tuple[Path, Path],
) -> None:
    root, config_path = synthetic_dataset
    config = load_dataset_config(config_path)
    with pytest.raises(ValueError, match="outside dataset"):
        ensure_output_path_is_safe(config, root / "generated.csv")
    with pytest.raises(ValueError, match="outside dataset"):
        write_audit_report(config, "train", output_root=root / "audit")


def test_strict_audit_records_then_fails(
    synthetic_dataset: tuple[Path, Path], tmp_path: Path
) -> None:
    _, config_path = synthetic_dataset
    raw = config_path.read_text(encoding="utf-8").replace(
        "fail_on_bad_files: false", "fail_on_bad_files: true"
    )
    strict_path = tmp_path / "strict.yaml"
    strict_path.write_text(raw, encoding="utf-8")
    config = load_dataset_config(strict_path)
    with pytest.raises(RuntimeError, match="strict data audit failed"):
        write_audit_report(config, "train")
    assert (config.audit.output_root / "train" / "bad_files.csv").is_file()


def test_strict_audit_fails_on_same_content_label_conflict(
    synthetic_dataset: tuple[Path, Path],
) -> None:
    root, config_path = synthetic_dataset
    manifest = root / "metadata" / "samples.csv"
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    duplicate = next(row for row in rows if row["issue"] == "exact_duplicate")
    duplicate["observed_label"] = "0005" if duplicate["observed_label"] != "0005" else "0004"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    config = load_dataset_config(config_path)
    with pytest.raises(RuntimeError, match="same-content label conflicts"):
        write_audit_report(config, "train")
