"""Build the deterministic long-tailed stress split from the current-stage train part.

Read-only with respect to official data: the script consumes the audited manifest
and the frozen split file, and never touches ``DataSetStage1/test``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.class_mapping import ClassMapping  # noqa: E402
from aictrack1.data.long_tail_split import (  # noqa: E402
    LONG_TAIL_EVAL,
    LONG_TAIL_TRAIN,
    LongTailSplitConfig,
    build_long_tail_split,
    read_split_assignments,
    split_duplicate_cluster_violations,
    summarize_partition,
    write_long_tail_split,
)
from aictrack1.data.manifest import read_manifest, sha256_file  # noqa: E402
from aictrack1.evaluation.long_tail import (  # noqa: E402
    HEAD,
    MID,
    TAIL,
    retention_diagnostics,
    retention_gate,
    stratify_classes,
    write_stratification,
)


def _load_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "long_tail_split" not in payload:
        raise ValueError("configuration requires a long_tail_split section")
    return payload["long_tail_split"]


def _resolve(config_path: Path, value: str) -> Path:
    """Resolve an input/output path relative to the configuration file."""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (config_path.parent / candidate).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = _load_config(config_path)
    inputs = config["inputs"]
    outputs = config["outputs"]
    manifest_path = _resolve(config_path, inputs["train_manifest"])
    split_path = _resolve(config_path, inputs["split"])
    class_mapping_path = _resolve(config_path, inputs["class_mapping"])
    class_counts_path = _resolve(config_path, inputs["reference_class_counts"])

    if "test" in manifest_path.as_posix().split("/"):
        raise ValueError("long-tail split builder refuses any path that looks like test data")

    records = read_manifest(manifest_path)
    assignments = read_split_assignments(split_path)
    if len(assignments) != len(records):
        raise ValueError("split file and manifest cover different sample counts")

    reference_counts = tuple(
        int(row["count"])
        for row in sorted(
            csv.DictReader(class_counts_path.open("r", encoding="utf-8-sig", newline="")),
            key=lambda row: str(row["official_class_id"]),
        )
    )
    mapping = ClassMapping.from_csv(class_mapping_path)
    num_classes = len(mapping)
    if len(reference_counts) != num_classes:
        raise ValueError("class counts and class mapping cover different class counts")

    stratification = stratify_classes(
        reference_counts,
        head_fraction=float(config["head_fraction"]),
        mid_fraction=float(config["mid_fraction"]),
        seed=int(config["seed"]),
    )
    split_config = LongTailSplitConfig(
        reference_partition=str(config["reference_partition"]),
        train_fraction=float(config["train_fraction"]),
        eval_fraction=float(config["eval_fraction"]),
        tail_retention=float(config["tail_retention"]),
        mid_retention=float(config["mid_retention"]),
        head_fraction=float(config["head_fraction"]),
        mid_fraction=float(config["mid_fraction"]),
        min_samples_per_class=int(config["min_samples_per_class"]),
        seed=int(config["seed"]),
    )
    assignment = build_long_tail_split(
        records,
        assignments,
        config=split_config,
        stratification_assignments=stratification.assignments,
        class_counts=reference_counts,
    )
    train_counts = summarize_partition(
        records, assignment, partition=LONG_TAIL_TRAIN, num_classes=num_classes
    )
    eval_counts = summarize_partition(
        records, assignment, partition=LONG_TAIL_EVAL, num_classes=num_classes
    )
    reference_partition_counts = summarize_partition(
        records, assignments, partition=str(config["reference_partition"]), num_classes=num_classes
    )
    retention = retention_diagnostics(
        reference_partition_counts, train_counts, stratification=stratification
    )
    gate = retention_gate(
        retention,
        min_class_rate=float(config["gates"]["min_per_class_rate"]),
        min_tail_to_head_ratio=float(config["gates"]["min_tail_to_head_rate_ratio"]),
    )

    split_sha256 = write_long_tail_split(assignment, _resolve(config_path, outputs["split"]))
    write_stratification(
        stratification,
        _resolve(config_path, outputs["stratification"]),
        retention_rate_by_class={
            index: (
                1.0
                if reference_partition_counts[index] == 0
                else train_counts[index] / reference_partition_counts[index]
            )
            for index in range(num_classes)
        },
    )
    summary = {
        "stage": str(config["stage"]),
        "scope": "current-stage train partition only; competition test data not accessed",
        "seed": int(config["seed"]),
        "reference_partition": str(config["reference_partition"]),
        "reference_class_total": sum(reference_counts),
        "num_classes": num_classes,
        "stratification": {
            "rule": "sort classes by descending reference count, tie-break by class index",
            "head_fraction": float(config["head_fraction"]),
            "mid_fraction": float(config["mid_fraction"]),
            "class_counts": stratification.counts,
        },
        "schedule": {
            "head_retention": 1.0,
            "mid_retention": float(config["mid_retention"]),
            "tail_retention": float(config["tail_retention"]),
            "min_samples_per_class": int(config["min_samples_per_class"]),
            "eval_fraction": float(config["eval_fraction"]),
        },
        "partitions": {
            LONG_TAIL_TRAIN: {
                "sample_count": sum(train_counts),
                "class_count": sum(1 for value in train_counts if value > 0),
                "class_count_distribution": _count_distribution(train_counts, stratification),
            },
            LONG_TAIL_EVAL: {
                "sample_count": sum(eval_counts),
                "class_count": sum(1 for value in eval_counts if value > 0),
                "class_count_distribution": _count_distribution(eval_counts, stratification),
            },
        },
        "duplicate_cluster_leakage_count": split_duplicate_cluster_violations(records, assignment),
        "retention": retention,
        "retention_gate": gate,
        "hashes": {
            "train_manifest": sha256_file(manifest_path),
            "split": sha256_file(split_path),
            "class_mapping_semantic": mapping.digest(),
            "class_counts": sha256_file(class_counts_path),
            "long_tail_split": split_sha256,
        },
    }
    summary_path = _resolve(config_path, outputs["summary"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not gate["passed"]:
        print("retention gate FAILED", file=sys.stderr)
        return 1
    return 0


def _count_distribution(
    counts: tuple[int, ...], stratification
) -> dict[str, dict[str, float | int]]:
    payload: dict[str, dict[str, float | int]] = {}
    for stratum in (HEAD, MID, TAIL):
        values = [
            counts[index]
            for index in range(stratification.num_classes)
            if stratification.assignments[index] == stratum
        ]
        if not values:
            continue
        payload[stratum] = {
            "min": min(values),
            "max": max(values),
            "total": sum(values),
        }
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
