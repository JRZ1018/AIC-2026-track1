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
                "transforms": {"image_size": 32, "resize_size": 32},
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


def test_competition_stage_requires_unlabeled_test_source(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "stage": "preliminary",
                "dataset_root": str(tmp_path / "data"),
                "class_mapping": "classes.csv",
                "extensions": ["png"],
                "sources": {
                    "train": {
                        "root": "train",
                        "label_source": "directory",
                        "class_from_parent": True,
                    }
                },
                "audit": {"output_root": str(tmp_path / "audit")},
                "split": {"validation_fraction": 0.2},
                "transforms": {"image_size": 32, "resize_size": 32},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unlabeled test"):
        load_dataset_config(path)


def test_boolean_contract_does_not_accept_truthy_strings(
    synthetic_dataset: tuple[Path, Path],
) -> None:
    _, config_path = synthetic_dataset
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["dataset"]["allow_truncated_images"] = "false"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a boolean"):
        load_dataset_config(config_path)


def test_derived_mapping_must_be_outside_immutable_data(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "train" / "0000").mkdir(parents=True)
    (data / "test").mkdir()
    path = tmp_path / "stage1.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "stage": "preliminary",
                "dataset_root": str(data),
                "class_mapping": "metadata/classes.csv",
                "class_mapping_provenance": "train_directory_names",
                "extensions": ["jpg"],
                "sources": {
                    "train": {
                        "root": "train",
                        "label_source": "directory",
                        "class_from_parent": True,
                    },
                    "test": {"root": "test", "label_source": "none"},
                },
                "audit": {"output_root": str(tmp_path / "audit")},
                "split": {"validation_fraction": 0.2},
                "transforms": {"image_size": 32, "resize_size": 32},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside the immutable"):
        load_dataset_config(path)
