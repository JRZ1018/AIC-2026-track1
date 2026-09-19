"""Derive a lossless class mapping from immutable training directory names."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.class_mapping import ClassMapping
from aictrack1.data.contracts import ensure_output_path_is_safe, load_dataset_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_dataset_config(args.config)
    if config.class_mapping_provenance != "train_directory_names":
        raise ValueError("configuration does not authorize directory-derived class mapping")
    output = ensure_output_path_is_safe(config, config.class_mapping)
    mapping = ClassMapping.from_train_directories(
        config.sources["train"].root,
        official_id_pattern=config.submission.official_class_id_pattern,
    )
    mapping.write_csv(output)
    print(f"classes={len(mapping)} sha256={mapping.digest()} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
