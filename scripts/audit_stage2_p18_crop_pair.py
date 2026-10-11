"""Pre-registered P18 crop-0.95 vs same-machine P13 fold2 four-view gate.

Reads Stage2 train-derived validation predictions only; never opens test data.
"""

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

from aictrack1.data.manifest import sha256_file  # noqa: E402
from aictrack1.evaluation.feature_geometry import grouped_bootstrap_delta  # noqa: E402

BASELINE_SHA = "633c323b877f9c2647704cd08c6a89931ea43c99bf15503f7edbf9a57311df80"
CANDIDATE_SHA = "fe2a21403ca6931fba7c168e0180579ae72a22c7cbe0fd4c912b4952a3359035"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "target", "four_view_prediction"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"four-view prediction columns missing: {path}")
        rows = list(reader)
    if len(rows) != 29737 or len({row["sample_id"] for row in rows}) != 29737:
        raise ValueError("P18 requires 29,737 unique train-derived validation rows")
    return rows


def _manifest_groups_and_conflicts(path: Path) -> tuple[dict[str, str], set[str]]:
    groups: dict[str, str] = {}
    hash_by_id: dict[str, str] = {}
    labels_by_hash: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "source_split", "near_duplicate_group", "sha256", "official_class_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Stage2 train manifest columns missing")
        for row in reader:
            sample_id = row["sample_id"]
            if row["source_split"] != "train" or sample_id in groups:
                raise ValueError("manifest is not unique Stage2 train only")
            groups[sample_id] = (
                f"near:{row['near_duplicate_group']}"
                if row["near_duplicate_group"] else f"singleton:{sample_id}"
            )
            hash_by_id[sample_id] = row["sha256"]
            labels_by_hash.setdefault(row["sha256"], set()).add(row["official_class_id"])
    if len(groups) != 148695:
        raise ValueError("incomplete Stage2 train manifest")
    conflicts = {sample_id for sample_id, digest in hash_by_id.items() if len(labels_by_hash[digest]) > 1}
    return groups, conflicts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnostics_root = (ROOT / "artifacts" / "diagnostics").resolve()
    reports_root = (ROOT / "artifacts" / "reports").resolve()
    baseline_dir, candidate_dir, output = (
        args.baseline_dir.resolve(), args.candidate_dir.resolve(), args.output.resolve()
    )
    if not baseline_dir.is_relative_to(diagnostics_root) or not candidate_dir.is_relative_to(diagnostics_root):
        raise ValueError("P18 gate refuses predictions outside local train-derived diagnostics")
    if not output.is_relative_to(reports_root) or output.exists():
        raise ValueError("P18 gate output must be a new path under artifacts/reports")
    baseline_report = json.loads((baseline_dir / "report.json").read_text(encoding="utf-8"))
    candidate_report = json.loads((candidate_dir / "report.json").read_text(encoding="utf-8"))
    for name, report, expected_sha in (
        ("baseline", baseline_report, BASELINE_SHA),
        ("candidate", candidate_report, CANDIDATE_SHA),
    ):
        if report.get("checkpoint_sha256") != expected_sha or report.get("smoke_only") is not False:
            raise ValueError(f"{name} checkpoint or full-run identity mismatch")
        if report.get("evaluated_count") != 29737 or "train-derived" not in report.get("scope", ""):
            raise ValueError(f"{name} is not full Stage2 train-derived validation")
        if report.get("four_view_resize") != 384:
            raise ValueError(f"{name} is not fixed scale384 four-view")
    for key in ("split_sha256", "manifest_sha256", "class_mapping_sha256", "four_view_protocol"):
        if baseline_report[key] != candidate_report[key]:
            raise ValueError(f"P18 paired source/protocol mismatch: {key}")
    manifest = ROOT / "artifacts" / "audits" / "stage2" / "train" / "samples.csv"
    if sha256_file(manifest) != baseline_report["manifest_sha256"]:
        raise ValueError("current Stage2 train manifest differs from diagnostic source")
    groups, conflicts = _manifest_groups_and_conflicts(manifest)
    baseline_path = baseline_dir / "paired_predictions.csv"
    candidate_path = candidate_dir / "paired_predictions.csv"
    baseline, candidate = _rows(baseline_path), _rows(candidate_path)
    first = np.empty(29737, dtype=np.float64)
    second = np.empty(29737, dtype=np.float64)
    ordinary = np.empty(29737, dtype=bool)
    group_ids: list[str] = []
    for index, (left, right) in enumerate(zip(baseline, candidate, strict=True)):
        sample_id = left["sample_id"]
        if sample_id != right["sample_id"] or left["target"] != right["target"] or sample_id not in groups:
            raise ValueError("P18 predictions are not aligned on Stage2 train validation IDs")
        first[index] = float(int(left["target"]) == int(left["four_view_prediction"]))
        second[index] = float(int(right["target"]) == int(right["four_view_prediction"]))
        ordinary[index] = sample_id not in conflicts
        group_ids.append(groups[sample_id])
    if not ordinary.any():
        raise ValueError("ordinary subset is empty")
    if abs(first.mean() - baseline_report["four_view"]["top1"]) > 1e-6:
        raise ValueError("baseline row accuracy differs from its report")
    if abs(second.mean() - candidate_report["four_view"]["top1"]) > 1e-6:
        raise ValueError("candidate row accuracy differs from its report")
    bootstrap = grouped_bootstrap_delta(first, second, group_ids, seed=20260925, replicates=500)
    macro_delta = candidate_report["four_view"]["macro_accuracy"] - baseline_report["four_view"]["macro_accuracy"]
    tail_delta = candidate_report["four_view"]["tail_class_mean_recall"] - baseline_report["four_view"]["tail_class_mean_recall"]
    ordinary_delta = float(second[ordinary].mean() - first[ordinary].mean())
    gate = {
        "top1_at_least_plus_0_30pp": bootstrap["delta"] >= 0.003,
        "group_ci95_lower_positive": bootstrap["ci95_low"] > 0,
        "macro_nonnegative": macro_delta >= 0,
        "tail_class_macro_at_least_minus_0_25pp": tail_delta >= -0.0025,
        "ordinary_nonconflict_nonnegative": ordinary_delta >= 0,
    }
    result = {
        "scope": "P18 fold2 vs same-machine P13 8ep; Stage2 train-derived validation only; no test",
        "baseline_checkpoint_sha256": BASELINE_SHA,
        "candidate_checkpoint_sha256": CANDIDATE_SHA,
        "baseline_report_sha256": sha256_file(baseline_dir / "report.json"),
        "candidate_report_sha256": sha256_file(candidate_dir / "report.json"),
        "baseline_predictions_sha256": sha256_file(baseline_path),
        "candidate_predictions_sha256": sha256_file(candidate_path),
        "manifest_sha256": sha256_file(manifest),
        "validation_count": 29737,
        "ordinary_count": int(ordinary.sum()),
        "top1": {"baseline": float(first.mean()), "candidate": float(second.mean()), "delta": bootstrap["delta"]},
        "macro_delta": float(macro_delta),
        "tail_class_macro_delta": float(tail_delta),
        "ordinary_nonconflict_delta": ordinary_delta,
        "paired_wins": int(((first == 0) & (second == 1)).sum()),
        "paired_losses": int(((first == 1) & (second == 0)).sum()),
        "near_duplicate_group_bootstrap": bootstrap,
        "gate": gate,
        "gate_passed": all(gate.values()),
        "caveat": "One reused noisy validation fold; passing would require separately approved fold3/full retraining before test inference",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"gate": gate, "gate_passed": result["gate_passed"], "top1_delta": bootstrap["delta"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
