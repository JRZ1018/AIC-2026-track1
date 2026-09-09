from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aictrack1.data.synthetic import SyntheticConfig, generate_synthetic_dataset


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "dataset"
    generate_synthetic_dataset(
        SyntheticConfig(
            output_root=dataset_root,
            seed=1234,
            num_classes=6,
            classes_per_family=3,
            max_samples_per_class=10,
            imbalance_ratio=2.0,
            image_size=40,
            noise_mode="family",
            noise_rate=0.35,
            exact_duplicate_count=2,
            near_duplicate_count=2,
            corrupt_count=1,
            truncated_count=1,
        )
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "stage": "synthetic",
                    "dataset_root": str(dataset_root),
                    "class_mapping": "metadata/classes.csv",
                    "extensions": [".png"],
                    "allow_truncated_images": False,
                    "sources": {
                        "train": {
                            "root": "images",
                            "label_source": "manifest",
                            "manifest": "metadata/samples.csv",
                        }
                    },
                    "audit": {
                        "output_root": str(tmp_path / "audit"),
                        "workers": 2,
                        "near_duplicate_hamming": 3,
                        "near_duplicate_max_bucket": 100,
                        "fail_on_bad_files": False,
                        "fail_on_label_conflicts": True,
                    },
                    "split": {"validation_fraction": 0.25, "seed": 7, "min_train_per_class": 2},
                    "transforms": {
                        "image_size": 40,
                        "resize_size": 40,
                        "train_crop_scale_min": 0.8,
                        "train_horizontal_flip_probability": 0.5,
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return dataset_root, config_path
