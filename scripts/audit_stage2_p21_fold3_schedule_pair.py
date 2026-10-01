"""P21 fold3 confirmation, using unchanged pre-registered fold2 gate definitions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml

from audit_stage2_p21_schedule_pair import groups_and_conflicts, rows, grouped_bootstrap_delta
from run_stage2_p21_fold3_diagnostics import ROOT, config, sha, training_evidence

MANIFEST = ROOT / "artifacts/audits/stage2/train/samples.csv"
SPLIT = ROOT / "artifacts/audits/stage2/train/split_fold3_seeded_random_20260925.csv"


def gate_values(bootstrap: dict, macro: float, tail: float, ordinary: float) -> dict:
    return {"top1_at_least_plus_0_10pp": bootstrap["delta"] >= 0.001,
            "group_ci95_lower_positive": bootstrap["ci95_low"] > 0, "macro_nonnegative": macro >= 0,
            "tail_class_macro_at_least_minus_0_25pp": tail >= -0.0025,
            "ordinary_nonconflict_nonnegative": ordinary >= 0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    control_dir, candidate_dir, output = (args.control_dir.resolve(), args.candidate_dir.resolve(), args.output.resolve())
    if any(not p.is_relative_to((ROOT / "artifacts/diagnostics").resolve()) for p in (control_dir, candidate_dir)):
        raise ValueError("only train-derived local diagnostics allowed")
    if not output.is_relative_to((ROOT / "artifacts/reports").resolve()) or output.exists():
        raise ValueError("gate must be a new report under artifacts/reports")
    evidence = training_evidence(safe_load=False)
    control_config = yaml.safe_load(config(14).read_text(encoding="utf-8"))["experiment"]
    candidate_config = yaml.safe_load(config(20).read_text(encoding="utf-8"))["experiment"]
    expected = {**candidate_config, "run_id": control_config["run_id"], "output_dir": control_config["output_dir"],
                "training": {**candidate_config["training"], "epochs": 14}}
    if control_config != expected or candidate_config["training"]["epochs"] != 20:
        raise ValueError("P21 fold3 configs differ beyond schedule/identity")
    reports = [json.loads((p / "report.json").read_text(encoding="utf-8")) for p in (control_dir, candidate_dir)]
    for n, report in zip((14, 20), reports, strict=True):
        if (report.get("checkpoint_sha256") != evidence["checks"][str(n)]["best_sha256"]
                or report.get("experiment_config_sha256") != sha(config(n))
                or report.get("smoke_only") is not False or report.get("evaluated_count") != 29737
                or report.get("full_validation_count") != 29737 or report.get("checkpoint_role") != "selected_best"
                or "train-derived" not in report.get("scope", "") or report.get("four_view_resize") != 384
                or report.get("split_sha256") != sha(SPLIT)):
            raise ValueError("P21 fold3 diagnostic provenance/completeness failed")
    left_report, right_report = reports
    for key in ("split_sha256", "manifest_sha256", "class_mapping_sha256", "four_view_protocol"):
        if left_report[key] != right_report[key]:
            raise ValueError(f"P21 fold3 reports differ in {key}")
    if sha(MANIFEST) != left_report["manifest_sha256"]:
        raise ValueError("Stage2 train manifest changed")
    group, conflicts = groups_and_conflicts()
    paths = [p / "paired_predictions.csv" for p in (control_dir, candidate_dir)]
    control, candidate = (rows(p) for p in paths)
    base, newer = np.empty(29737), np.empty(29737)
    ordinary = np.empty(29737, dtype=bool)
    ids = []
    for i, (left, right) in enumerate(zip(control, candidate, strict=True)):
        sid = left["sample_id"]
        if sid != right["sample_id"] or left["target"] != right["target"] or sid not in group:
            raise ValueError("P21 fold3 prediction rows do not align")
        base[i], newer[i] = left["target"] == left["four_view_prediction"], right["target"] == right["four_view_prediction"]
        ordinary[i] = sid not in conflicts
        ids.append(group[sid])
    if not ordinary.any() or any(abs(a.mean() - r["four_view"]["top1"]) > 1e-6
                                for a, r in zip((base, newer), reports, strict=True)):
        raise ValueError("P21 fold3 row accuracy/report mismatch or empty ordinary subset")
    bootstrap = grouped_bootstrap_delta(base, newer, ids, seed=20260925, replicates=500)
    macro = right_report["four_view"]["macro_accuracy"] - left_report["four_view"]["macro_accuracy"]
    tail = right_report["four_view"]["tail_class_mean_recall"] - left_report["four_view"]["tail_class_mean_recall"]
    ordinary_delta = float(newer[ordinary].mean() - base[ordinary].mean())
    gate = gate_values(bootstrap, macro, tail, ordinary_delta)
    result = {"scope": "P21 same-source fold3 20 vs 14; Stage2 train-derived validation; no test",
        "control_checkpoint_sha256": left_report["checkpoint_sha256"],
        "candidate_checkpoint_sha256": right_report["checkpoint_sha256"],
        "config_sha256": {str(n): sha(config(n)) for n in (14, 20)},
        "report_sha256": {str(n): sha(p / "report.json") for n, p in zip((14, 20), (control_dir, candidate_dir))},
        "predictions_sha256": {str(n): sha(p) for n, p in zip((14, 20), paths)},
        "manifest_sha256": sha(MANIFEST), "split_sha256": sha(SPLIT), "validation_count": 29737,
        "ordinary_count": int(ordinary.sum()), "top1": {"control": float(base.mean()),
            "candidate": float(newer.mean()), "delta": bootstrap["delta"]},
        "macro_delta": float(macro), "tail_class_macro_delta": float(tail),
        "ordinary_nonconflict_delta": ordinary_delta, "near_duplicate_group_bootstrap": bootstrap,
        "paired_wins": int(((base == 0) & (newer == 1)).sum()),
        "paired_losses": int(((base == 1) & (newer == 0)).sum()), "gate": gate, "gate_passed": all(gate.values()),
        "caveat": "Previously used noisy folds overlap; this does not measure seed variance or guarantee hidden-test improvement"}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"gate_passed": result["gate_passed"], "top1_delta": bootstrap["delta"], "gate": gate}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
