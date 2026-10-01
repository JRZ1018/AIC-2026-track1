"""Pre-registered P21 20-vs-14 epoch gate on paired Stage2 train validation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aictrack1.data.manifest import sha256_file  # noqa: E402
from aictrack1.evaluation.feature_geometry import grouped_bootstrap_delta  # noqa: E402

CANDIDATE_CONFIG = ROOT / "configs/experiment/stage2_p21_fold2_sce_ac_20ep_4090_screen.yaml"
CONTROL_CONFIG = ROOT / "configs/experiment/stage2_p21_fold2_sce_ac_14ep_4090_matched.yaml"
CANDIDATE_SHA = "927d0d2613e071decd3a3e6f9303828eaca8f66a3296ab7a01ce16446681dc67"
MANIFEST = ROOT / "artifacts/audits/stage2/train/samples.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"sample_id", "target", "four_view_prediction"}.issubset(reader.fieldnames):
            raise ValueError("P21 paired predictions lack required columns")
        result = list(reader)
    if len(result) != 29737 or len({r["sample_id"] for r in result}) != 29737:
        raise ValueError("P21 requires all 29,737 unique validation predictions")
    return result


def groups_and_conflicts() -> tuple[dict[str, str], set[str]]:
    group: dict[str, str] = {}
    hash_by_id: dict[str, str] = {}
    labels_by_hash: dict[str, set[str]] = {}
    with MANIFEST.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"sample_id", "source_split", "near_duplicate_group", "sha256", "official_class_id"}.issubset(reader.fieldnames):
            raise ValueError("Stage2 train manifest columns missing")
        for row in reader:
            sid = row["sample_id"]
            if row["source_split"] != "train" or sid in group:
                raise ValueError("Stage2 train manifest is not unique train-only")
            group[sid] = f"near:{row['near_duplicate_group']}" if row["near_duplicate_group"] else f"singleton:{sid}"
            hash_by_id[sid] = row["sha256"]
            labels_by_hash.setdefault(row["sha256"], set()).add(row["official_class_id"])
    if len(group) != 148695:
        raise ValueError("Stage2 train manifest is incomplete")
    conflicts = {sid for sid, digest in hash_by_id.items() if len(labels_by_hash[digest]) > 1}
    return group, conflicts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnostics = (ROOT / "artifacts/diagnostics").resolve()
    reports = (ROOT / "artifacts/reports").resolve()
    control_dir, candidate_dir, output = (args.control_dir.resolve(), args.candidate_dir.resolve(), args.output.resolve())
    if not control_dir.is_relative_to(diagnostics) or not candidate_dir.is_relative_to(diagnostics):
        raise ValueError("P21 gate accepts only local train-derived diagnostics")
    if not output.is_relative_to(reports) or output.exists():
        raise ValueError("P21 gate report must be a new file under artifacts/reports")
    control_config = yaml.safe_load(CONTROL_CONFIG.read_text(encoding="utf-8"))["experiment"]
    candidate_config = yaml.safe_load(CANDIDATE_CONFIG.read_text(encoding="utf-8"))["experiment"]
    expected = dict(candidate_config)
    expected.update({"run_id": control_config["run_id"], "output_dir": control_config["output_dir"],
                     "training": {**candidate_config["training"], "epochs": 14}})
    if control_config != expected or candidate_config["training"]["epochs"] != 20:
        raise ValueError("P21 control and candidate configs differ beyond schedule/identity")
    control_report = json.loads((control_dir / "report.json").read_text(encoding="utf-8"))
    candidate_report = json.loads((candidate_dir / "report.json").read_text(encoding="utf-8"))
    if candidate_report.get("checkpoint_sha256") != CANDIDATE_SHA:
        raise ValueError("P21 twenty-epoch candidate checkpoint changed")
    for name, report, config in (("control", control_report, CONTROL_CONFIG),
                                 ("candidate", candidate_report, CANDIDATE_CONFIG)):
        if (report.get("experiment_config_sha256") != sha256_file(config)
                or report.get("smoke_only") is not False or report.get("evaluated_count") != 29737
                or "train-derived" not in report.get("scope", "") or report.get("four_view_resize") != 384):
            raise ValueError(f"{name} P21 report provenance/completeness failed")
    for key in ("split_sha256", "manifest_sha256", "class_mapping_sha256", "four_view_protocol"):
        if control_report[key] != candidate_report[key]:
            raise ValueError(f"P21 reports differ in {key}")
    if sha256_file(MANIFEST) != control_report["manifest_sha256"]:
        raise ValueError("current Stage2 train manifest differs from P21 report")
    group, conflicts = groups_and_conflicts()
    control_path, candidate_path = control_dir / "paired_predictions.csv", candidate_dir / "paired_predictions.csv"
    control, candidate = rows(control_path), rows(candidate_path)
    base = np.empty(29737, dtype=np.float64)
    newer = np.empty(29737, dtype=np.float64)
    ordinary = np.empty(29737, dtype=bool)
    ids: list[str] = []
    for index, (left, right) in enumerate(zip(control, candidate, strict=True)):
        sid = left["sample_id"]
        if sid != right["sample_id"] or left["target"] != right["target"] or sid not in group:
            raise ValueError("P21 paired validation rows do not align")
        base[index] = float(left["target"] == left["four_view_prediction"])
        newer[index] = float(right["target"] == right["four_view_prediction"])
        ordinary[index] = sid not in conflicts
        ids.append(group[sid])
    if not ordinary.any():
        raise ValueError("P21 ordinary non-conflict subset empty")
    if abs(base.mean() - control_report["four_view"]["top1"]) > 1e-6 or abs(newer.mean() - candidate_report["four_view"]["top1"]) > 1e-6:
        raise ValueError("P21 row accuracy disagrees with diagnostic report")
    bootstrap = grouped_bootstrap_delta(base, newer, ids, seed=20260925, replicates=500)
    macro = candidate_report["four_view"]["macro_accuracy"] - control_report["four_view"]["macro_accuracy"]
    tail = candidate_report["four_view"]["tail_class_mean_recall"] - control_report["four_view"]["tail_class_mean_recall"]
    ordinary_delta = float(newer[ordinary].mean() - base[ordinary].mean())
    gate = {"top1_at_least_plus_0_10pp": bootstrap["delta"] >= 0.001,
            "group_ci95_lower_positive": bootstrap["ci95_low"] > 0,
            "macro_nonnegative": macro >= 0,
            "tail_class_macro_at_least_minus_0_25pp": tail >= -0.0025,
            "ordinary_nonconflict_nonnegative": ordinary_delta >= 0}
    result = {"scope": "P21 same-source fold2 20 vs 14 epochs; Stage2 train-derived validation; no test",
              "control_checkpoint_sha256": control_report["checkpoint_sha256"],
              "candidate_checkpoint_sha256": CANDIDATE_SHA,
              "control_config_sha256": sha256_file(CONTROL_CONFIG),
              "candidate_config_sha256": sha256_file(CANDIDATE_CONFIG),
              "control_report_sha256": sha256_file(control_dir / "report.json"),
              "candidate_report_sha256": sha256_file(candidate_dir / "report.json"),
              "control_predictions_sha256": sha256_file(control_path),
              "candidate_predictions_sha256": sha256_file(candidate_path),
              "manifest_sha256": sha256_file(MANIFEST), "validation_count": 29737,
              "ordinary_count": int(ordinary.sum()), "top1": {"control": float(base.mean()),
              "candidate": float(newer.mean()), "delta": bootstrap["delta"]},
              "macro_delta": float(macro), "tail_class_macro_delta": float(tail),
              "ordinary_nonconflict_delta": ordinary_delta,
              "paired_wins": int(((base == 0) & (newer == 1)).sum()),
              "paired_losses": int(((base == 1) & (newer == 0)).sum()),
              "near_duplicate_group_bootstrap": bootstrap, "gate": gate, "gate_passed": all(gate.values()),
              "caveat": "Fixed noisy fold and overlapping train-derived splits do not quantify seed variance or hidden-test generalization"}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"gate_passed": result["gate_passed"], "top1_delta": bootstrap["delta"], "gate": gate}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
