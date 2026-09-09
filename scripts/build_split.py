"""Create a deterministic validation split from an audited manifest."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.contracts import ensure_output_path_is_safe, load_dataset_config
from aictrack1.data.manifest import read_manifest
from aictrack1.data.splits import build_grouped_stratified_split, write_split


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = load_dataset_config(args.config)
    records = read_manifest(args.manifest)
    allowed_statuses = {"ok"}
    if config.allow_truncated_images:
        allowed_statuses.add("truncated")
    assignment = build_grouped_stratified_split(
        records,
        validation_fraction=config.split.validation_fraction,
        seed=config.split.seed,
        min_train_per_class=config.split.min_train_per_class,
        allowed_decode_statuses=frozenset(allowed_statuses),
    )
    output = ensure_output_path_is_safe(config, args.output)
    digest = write_split(assignment, output)
    counts = Counter(assignment.values())
    print(f"train={counts['train']} validation={counts['validation']} excluded_bad_file={counts['excluded_bad_file']} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
