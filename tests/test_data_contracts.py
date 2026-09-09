from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aictrack1.data.contracts import load_dataset_config


def test_explicit_synthetic_contract_loads(synthetic_dataset: tuple[Path, Path]) -> None:
    root, config_path = synthetic_dataset
    config = load_dataset_config(config_path)
    assert config.stage == "synthetic"
    assert config.dataset_root == root.resolve()
    assert config.sources["train"].label_source == "manifest"


def test_test_source_cannot_have_labels(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "stage": "synthetic",
                "dataset_root": str(tmp_path),
                "class_mapping": "classes.csv",
                "extensions": ["png"],
                "sources": {"train": {"root": "train", "label_source": "directory", "class_from_parent": True}, "test": {"root": "test", "label_source": "directory", "class_from_parent": True}},
                "audit": {"output_root": str(tmp_path.parent / "audit")},
                "split": {"validation_fraction": 0.2},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="test source"):
        load_dataset_config(path)


def test_output_cannot_be_inside_source(synthetic_dataset: tuple[Path, Path]) -> None:
    root, config_path = synthetic_dataset
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["dataset"]["audit"]["output_root"] = str(root / "images" / "audit")
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="inside dataset_root"):
        load_dataset_config(config_path)
