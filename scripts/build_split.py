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
from aictrack1.data.long_tail_split import read_split_assignments
from aictrack1.data.splits import build_grouped_stratified_split, leakage_group, write_split


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--group-order",
        choices=("largest_first", "seeded_random"),
        default="largest_first",
        help="Keep historical default unless explicitly building a diagnostic fold",
    )
    parser.add_argument(
        "--exclude-validation-from",
        type=Path,
        action="append",
        default=[],
        help="Keep validation groups from an earlier split out of this split's validation partition",
    )
    args = parser.parse_args()
    config = load_dataset_config(args.config)
    records = read_manifest(args.manifest)
    sample_to_group = {record.sample_id: leakage_group(record) for record in records}
    forbidden_groups: set[str] = set()
    for prior_path in args.exclude_validation_from:
        prior = read_split_assignments(prior_path)
        if set(prior) != set(sample_to_group):
            missing = len(set(sample_to_group) - set(prior))
            extra = len(set(prior) - set(sample_to_group))
            raise ValueError(f"prior split does not match manifest: {prior_path}; missing={missing} extra={extra}")
        forbidden_groups.update(sample_to_group[sample_id] for sample_id, partition in prior.items() if partition == "validation")
    allowed_statuses = {"ok"}
    if config.allow_truncated_images:
        allowed_statuses.add("truncated")
    assignment = build_grouped_stratified_split(
        records,
        validation_fraction=config.split.validation_fraction,
        seed=config.split.seed,
        min_train_per_class=config.split.min_train_per_class,
        allowed_decode_statuses=frozenset(allowed_statuses),
        group_order=args.group_order,
        forbidden_validation_groups=frozenset(forbidden_groups),
    )
    output = ensure_output_path_is_safe(config, args.output)
    digest = write_split(assignment, output)
    counts = Counter(assignment.values())
    print(f"train={counts['train']} validation={counts['validation']} excluded_bad_file={counts['excluded_bad_file']} forbidden_groups={len(forbidden_groups)} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
