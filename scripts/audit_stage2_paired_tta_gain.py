"""Cluster-aware paired accuracy audit on Stage2 train-derived validation only."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.evaluation.feature_geometry import grouped_bootstrap_delta  # noqa: E402
from aictrack1.data.manifest import sha256_file  # noqa: E402


def _read_predictions(path: Path, column: str) -> list[dict[str, str]]:
    if not path.is_relative_to((ROOT / "artifacts" / "diagnostics").resolve()):
        raise ValueError("predictions must come from train-derived diagnostic artifacts")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"sample_id", "target", column}.issubset(reader.fieldnames):
            raise ValueError("paired prediction columns are missing")
        rows = list(reader)
    if len(rows) != 29737 or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("expected one complete, unique Stage2 validation fold")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions", required=True, type=Path)
    parser.add_argument("--candidate-predictions", required=True, type=Path)
    parser.add_argument("--baseline-column", default="flip_tta_prediction")
    parser.add_argument("--candidate-column", default="flip_tta_prediction")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--replicates", type=int, default=500)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "artifacts" / "reports").resolve()):
        raise ValueError("audit report must be under artifacts/reports")
    if output.exists():
        raise FileExistsError(f"audit output already exists: {output}")
    train_manifest = ROOT / "artifacts" / "audits" / "stage2" / "train" / "samples.csv"
    with train_manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"sample_id", "source_split", "near_duplicate_group"}.issubset(reader.fieldnames):
            raise ValueError("Stage2 train manifest columns are missing")
        groups = {}
        for row in reader:
            if row["source_split"] != "train":
                raise ValueError("non-train record in Stage2 train manifest")
            if row["sample_id"] in groups:
                raise ValueError("duplicate Stage2 train sample ID")
            groups[row["sample_id"]] = (
                f"near:{row['near_duplicate_group']}"
                if row["near_duplicate_group"]
                else f"singleton:{row['sample_id']}"
            )
    baseline_path = args.baseline_predictions.resolve()
    candidate_path = args.candidate_predictions.resolve()
    baseline = _read_predictions(baseline_path, args.baseline_column)
    candidate = _read_predictions(candidate_path, args.candidate_column)
    base_correct = np.empty(len(baseline), dtype=np.float64)
    candidate_correct = np.empty(len(candidate), dtype=np.float64)
    group_ids = []
    for index, (first, second) in enumerate(zip(baseline, candidate, strict=True)):
        if first["sample_id"] != second["sample_id"] or first["target"] != second["target"]:
            raise ValueError("paired predictions have different sample or target order")
        sample_id = first["sample_id"]
        if sample_id not in groups:
            raise ValueError("prediction sample is not in Stage2 train manifest")
        base_correct[index] = float(first["target"] == first[args.baseline_column])
        candidate_correct[index] = float(second["target"] == second[args.candidate_column])
        group_ids.append(groups[sample_id])
    bootstrap = grouped_bootstrap_delta(
        base_correct, candidate_correct, group_ids,
        seed=args.seed, replicates=args.replicates,
    )
    report = {
        "scope": "Stage2 train-derived fixed validation only; no test access",
        "baseline_predictions_sha256": sha256_file(baseline_path),
        "candidate_predictions_sha256": sha256_file(candidate_path),
        "train_manifest_sha256": sha256_file(train_manifest),
        "baseline_column": args.baseline_column,
        "candidate_column": args.candidate_column,
        "count": len(baseline),
        "candidate_wins": int(((candidate_correct == 1) & (base_correct == 0)).sum()),
        "baseline_wins": int(((base_correct == 1) & (candidate_correct == 0)).sum()),
        "paired_near_duplicate_group_bootstrap": bootstrap,
        "caveat": "Conditional on fixed fold models and noisy observed labels; does not capture retraining variance or hidden test shift",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
