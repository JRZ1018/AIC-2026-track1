"""Reconcile Stage 2 experiment-index scores with original per-run metrics.

This reads lightweight training artifacts only. It never opens competition test
data, checkpoints, or predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _selected_validation(metrics: dict) -> tuple[int, dict] | None:
    history = metrics.get("history", [])
    if not history or not any("validation" in epoch or "top1" in epoch for epoch in history):
        return None
    if "validation" in history[0]:
        selected = max(history, key=lambda epoch: epoch["validation"]["all"]["top1"])
        return int(selected["epoch"]), selected["validation"]
    selected = max(history, key=lambda epoch: epoch["top1"])
    return int(selected["epoch"]), selected


def audit(index_path: Path, root: Path) -> dict:
    rows = list(csv.DictReader(index_path.open("r", encoding="utf-8-sig", newline="")))
    report: dict = {"scope": "Stage 2 experiment index and original metrics only", "runs": [], "mismatches": []}
    for row in rows:
        if row["stage"] != "stage2":
            continue
        path = root / row["result_dir"] / "metrics.json"
        if not path.is_file():
            report["mismatches"].append({"run_id": row["run_id"], "field": "metrics.json", "reason": "missing"})
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        provenance = metrics.get("checkpoint_provenance", {})
        selected = _selected_validation(metrics)
        entry = {
            "run_id": row["run_id"],
            "training_scope": metrics.get("training_scope"),
            "indexed_top1": row["top1"] or None,
            "best_top1": metrics.get("best_top1"),
            "selected_epoch": selected[0] if selected else None,
            "split_sha256": provenance.get("split_sha256", metrics.get("split_sha256")),
        }
        if selected:
            validation = selected[1]
            if "all" in validation:
                entry.update(
                    top1=validation["all"]["top1"],
                    macro=validation["all"]["macro_accuracy"],
                    tail_sample_weighted=validation["long_tail"]["strata"]["tail"]["accuracy"],
                    tail_class_mean=validation["long_tail"]["strata"]["tail"]["evaluated_class_mean_accuracy"],
                    exact_conflict_sample_count=validation.get("conflict_sample_count"),
                )
            else:
                entry.update(top1=validation["top1"], macro=validation["macro_accuracy"])
            for field, key in (("top1", "top1"), ("macro_accuracy", "macro"), ("tail_accuracy", "tail_sample_weighted")):
                if row[field] and key in entry and abs(float(row[field]) - entry[key]) > 1e-7:
                    report["mismatches"].append({"run_id": row["run_id"], "field": field, "index": row[field], "source": entry[key]})
            if abs(float(metrics["best_top1"]) - entry["top1"]) > 1e-7:
                report["mismatches"].append({"run_id": row["run_id"], "field": "best_top1", "reason": "does not match selected history"})
        report["runs"].append(entry)
    report["run_count"] = len(report["runs"])
    report["mismatch_count"] = len(report["mismatches"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("experiments/index.csv"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.index, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Stage 2 runs={report['run_count']} mismatches={report['mismatch_count']}")
    for mismatch in report["mismatches"]:
        print(json.dumps(mismatch, sort_keys=True))
    return int(report["mismatch_count"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
