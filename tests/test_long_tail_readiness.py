from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from aictrack1.data.class_mapping import ClassEntry, ClassMapping
from aictrack1.data.manifest import SampleRecord, write_manifest
from aictrack1.evaluation.long_tail_readiness import (
    combined_long_tail_decision,
    effective_training_quality_profile,
    support_distribution_profile,
)


ROOT = Path(__file__).resolve().parents[1]


def test_support_distribution_balanced_is_l0_and_deterministic() -> None:
    first = support_distribution_profile([161, 165, 166, 168, 170])
    second = support_distribution_profile([161, 165, 166, 168, 170])
    assert first == second
    assert first["decision"] == "L0"
    assert first["support"]["effective_number"] == pytest.approx(4.998, abs=0.01)
    assert first["zero_sample_classes"] == []


def test_support_distribution_mild_tail_is_l1() -> None:
    report = support_distribution_profile([50] * 8 + [20] * 2)
    assert report["decision"] == "L1"
    assert any(item["metric"] == "min_to_median" for item in report["triggers"])


def test_support_distribution_severe_tail_and_zero_class_are_l2() -> None:
    report = support_distribution_profile([100] * 8 + [5, 0])
    assert report["decision"] == "L2"
    assert report["zero_sample_classes"] == [9]
    assert 9 in report["unsafe_split_classes"]


def test_effective_quality_distinguishes_raw_support_from_weight_collapse() -> None:
    labels = [0, 0, 1, 1, 2, 2]
    quality = effective_training_quality_profile(
        labels,
        [1.0] * 6,
        [1.0, 1.0, 0.5, 0.5, 0.02, 0.02],
        num_classes=3,
        recalls=[1.0, 0.5, 0.0],
    )
    assert quality["decision"] == "L2"
    assert quality["classes"][2]["raw_support"] == 2
    assert quality["classes"][2]["effective_quality"] == pytest.approx(0.04)
    assert quality["classes"][2]["effective_to_raw_ratio"] == pytest.approx(0.02)
    assert quality["summary"]["zero_recall_class_count"] == 1
    support = support_distribution_profile([2, 2, 2])
    assert support["decision"] == "L0"
    assert combined_long_tail_decision(support, quality) == "L2"


def test_effective_quality_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        effective_training_quality_profile([0], [], [1.0], num_classes=1)
    with pytest.raises(ValueError, match="between zero and one"):
        effective_training_quality_profile([0], [1.0], [1.1], num_classes=1)


def test_report_command_is_byte_deterministic(tmp_path: Path) -> None:
    mapping = ClassMapping(
        [
            ClassEntry("class-a", 0),
            ClassEntry("class-b", 1),
            ClassEntry("class-c", 2),
        ]
    )
    mapping_path = tmp_path / "classes.csv"
    mapping.write_csv(mapping_path)
    records = []
    for class_index, count in enumerate((5, 5, 5)):
        for sample_index in range(count):
            records.append(
                SampleRecord(
                    sample_id=f"{class_index}-{sample_index}",
                    relative_path=f"class-{class_index}/{sample_index}.jpg",
                    source_split="train",
                    observed_label=f"class-{chr(97 + class_index)}",
                    internal_class_index=class_index,
                    official_class_id=f"class-{chr(97 + class_index)}",
                )
            )
    manifest = tmp_path / "samples.csv"
    write_manifest(records, manifest)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    command = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        str(ROOT / "scripts" / "report_long_tail_readiness.py"),
        "--manifest",
        str(manifest),
        "--class-mapping",
        str(mapping_path),
        "--output-json",
    ]
    python = command[0] if Path(command[0]).is_file() else "D:/aicomp/python312/python.exe"
    command[0] = python
    subprocess.run(command + [str(first)], cwd=ROOT, check=True, capture_output=True, text=True)
    subprocess.run(command + [str(second)], cwd=ROOT, check=True, capture_output=True, text=True)
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text(encoding="utf-8"))["decision"] == "L0"
