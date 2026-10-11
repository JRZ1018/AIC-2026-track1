"""Combine the two frozen P19 train-derived head screens without test access."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.manifest import sha256_file  # noqa: E402


def _load(path: Path, fold: int) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("fold") != fold or report.get("validation_count") != 29737:
        raise ValueError(f"fold{fold} report identity or row count mismatch")
    if "train-derived" not in str(report.get("scope", "")) or "no test" not in str(report.get("scope", "")):
        raise ValueError("P19 aggregation refuses non-train-derived inputs")
    if report["raw_center_paired_near_duplicate_group_bootstrap"]["replicates"] != 500:
        raise ValueError("P19 screen must use 500 group bootstrap replicates")
    return report


def _combined_group_bootstrap(grouped: dict[str, list[int]]) -> dict[str, float | int]:
    values = np.array(list(grouped.values()), dtype=np.int64)
    counts = values[:, 0]
    differences = values[:, 2] - values[:, 1]
    rng = np.random.default_rng(20260925)
    draws = np.empty(500, dtype=np.float64)
    for index in range(500):
        sampled = rng.integers(0, len(values), size=len(values))
        draws[index] = differences[sampled].sum() / counts[sampled].sum()
    return {
        "delta": float(differences.sum() / counts.sum()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "group_count": len(values),
        "observation_count": int(counts.sum()),
        "replicates": 500,
        "seed": 20260925,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold2", required=True, type=Path)
    parser.add_argument("--fold3", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = [args.fold2.resolve(), args.fold3.resolve()]
    output = args.output.resolve()
    reports_root = (ROOT / "artifacts" / "reports").resolve()
    if any(not path.is_relative_to(reports_root) for path in paths):
        raise ValueError("P19 fold reports must be local artifacts/reports")
    if not output.is_relative_to(reports_root) or output.exists():
        raise ValueError("P19 aggregate output must be new under artifacts/reports")
    reports = [_load(paths[0], 2), _load(paths[1], 3)]
    if reports[0]["manifest_sha256"] != reports[1]["manifest_sha256"]:
        raise ValueError("P19 folds have different Stage2 train manifests")
    if reports[0]["class_mapping_sha256"] != reports[1]["class_mapping_sha256"]:
        raise ValueError("P19 folds have different class mappings")
    grouped: dict[str, list[int]] = {}
    deltas: dict[str, list[float]] = {key: [] for key in ("top1", "macro", "tail_class_macro", "ordinary_top1")}
    for report in reports:
        row_count = 0
        for row in report["raw_center_group_sufficient_statistics"]:
            group_id = str(row["group_id"])
            count = int(row["count"])
            first = int(row["baseline_correct"])
            second = int(row["candidate_correct"])
            if count < 1 or not 0 <= first <= count or not 0 <= second <= count:
                raise ValueError("invalid P19 group sufficient statistics")
            entry = grouped.setdefault(group_id, [0, 0, 0])
            entry[0] += count
            entry[1] += first
            entry[2] += second
            row_count += count
        if row_count != 29737:
            raise ValueError("P19 fold group statistics do not cover 29,737 samples")
        baseline = report["models"]["baseline"]["raw_center"]
        candidate = report["models"]["candidate"]["raw_center"]
        for key in deltas:
            deltas[key].append(float(candidate[key]) - float(baseline[key]))
    combined = _combined_group_bootstrap(grouped)
    means = {key: float(np.mean(values)) for key, values in deltas.items()}
    gate = {
        "fold_top1_nonnegative": all(value >= 0 for value in deltas["top1"]),
        "fold_mean_top1_at_least_0_20pp": means["top1"] >= 0.002,
        "fold_mean_macro_positive": means["macro"] > 0,
        "fold_mean_tail_class_macro_at_least_minus_0_25pp": means["tail_class_macro"] >= -0.0025,
        "fold_mean_ordinary_at_least_minus_0_25pp": means["ordinary_top1"] >= -0.0025,
        "combined_cluster_ci95_lower_positive": combined["ci95_low"] > 0,
    }
    result = {
        "scope": "P19 frozen-head Stage2 train-derived validation only; no test access",
        "fold_report_sha256": {"fold2": sha256_file(paths[0]), "fold3": sha256_file(paths[1])},
        "per_fold_delta_fraction": deltas,
        "fold_mean_delta_fraction": means,
        "combined_group_bootstrap": combined,
        "gate": gate,
        "gate_passed": all(gate.values()),
        "caveat": "Fold validation groups overlap across folds; bootstrap groups shared IDs together but does not measure retraining variance or hidden-test shift. Passing only authorizes a separately budgeted PEFT test.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"gate_passed": result["gate_passed"], "gate": gate}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
