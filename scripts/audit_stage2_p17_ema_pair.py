"""Pre-registered P17 EMA versus same-trajectory raw, train-only fold2 gate."""

from __future__ import annotations

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

RAW_SHA = "dbb1359c16d9ba567c935a8447a8f06bb2e29ae1347386a4a5dfa68a307fe5d2"
EMA_SHA = "3df2942e3d4a30f38c1cc92ee4e693ad8002c3009be7850829b7c780377b6b97"
RAW = ROOT / "artifacts/diagnostics/20261001-p17-raw-local-fourview384"
EMA = ROOT / "artifacts/diagnostics/20261001-p17-ema-local-fourview384"
OUTPUT = ROOT / "artifacts/reports/20261001-p17-ema-vs-own-raw-fold2.json"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {
            "sample_id", "target", "four_view_prediction"
        }.issubset(reader.fieldnames):
            raise ValueError("missing four-view prediction fields")
        rows = list(reader)
    if len(rows) != 29737 or len({r["sample_id"] for r in rows}) != 29737:
        raise ValueError("P17 requires all 29,737 unique fold2 validation rows")
    return rows


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    reports = [json.loads((folder / "report.json").read_text(encoding="utf-8"))
               for folder in (RAW, EMA)]
    for report, expected in zip(reports, (RAW_SHA, EMA_SHA), strict=True):
        if (report.get("checkpoint_sha256") != expected
                or report.get("evaluated_count") != 29737
                or report.get("smoke_only") is not False
                or report.get("four_view_resize") != 384
                or "train-derived" not in report.get("scope", "")):
            raise ValueError("P17 checkpoint, scope, or four-view protocol mismatch")
    for key in ("split_sha256", "manifest_sha256", "class_mapping_sha256",
                "experiment_config_sha256", "four_view_protocol"):
        if reports[0][key] != reports[1][key]:
            raise ValueError("P17 source mismatch: " + key)
    manifest = ROOT / "artifacts/audits/stage2/train/samples.csv"
    if sha256_file(manifest) != reports[0]["manifest_sha256"]:
        raise ValueError("train manifest SHA mismatch")
    groups: dict[str, str] = {}
    labels_by_hash: dict[str, set[str]] = {}
    hashes_by_id: dict[str, str] = {}
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = row["sample_id"]
            if row["source_split"] != "train" or sample_id in groups:
                raise ValueError("manifest must be unique Stage2 train only")
            groups[sample_id] = (
                "near:" + row["near_duplicate_group"] if row["near_duplicate_group"]
                else "singleton:" + sample_id
            )
            digest = row["sha256"]
            hashes_by_id[sample_id] = digest
            labels_by_hash.setdefault(digest, set()).add(row["official_class_id"])
    if len(groups) != 148695:
        raise ValueError("incomplete Stage2 train manifest")
    conflicts = {sample_id for sample_id, digest in hashes_by_id.items()
                 if len(labels_by_hash[digest]) > 1}
    paths = [folder / "paired_predictions.csv" for folder in (RAW, EMA)]
    raw, ema = [read_rows(path) for path in paths]
    correct = np.empty((2, 29737), dtype=np.float64)
    ordinary = np.empty(29737, dtype=bool)
    paired_groups: list[str] = []
    for index, (first, second) in enumerate(zip(raw, ema, strict=True)):
        sample_id = first["sample_id"]
        if (sample_id != second["sample_id"] or first["target"] != second["target"]
                or sample_id not in groups):
            raise ValueError("P17 raw/EMA predictions are not paired")
        correct[0, index] = float(first["target"] == first["four_view_prediction"])
        correct[1, index] = float(second["target"] == second["four_view_prediction"])
        ordinary[index] = sample_id not in conflicts
        paired_groups.append(groups[sample_id])
    if not ordinary.any():
        raise ValueError("ordinary group is empty")
    for index in range(2):
        if abs(correct[index].mean() - reports[index]["four_view"]["top1"]) > 1e-6:
            raise ValueError("prediction rows do not match diagnostic accuracy")
    bootstrap = grouped_bootstrap_delta(correct[0], correct[1], paired_groups,
                                        seed=20260925, replicates=500)
    macro_delta = reports[1]["four_view"]["macro_accuracy"] - reports[0]["four_view"]["macro_accuracy"]
    tail_delta = reports[1]["four_view"]["tail_class_mean_recall"] - reports[0]["four_view"]["tail_class_mean_recall"]
    ordinary_delta = float(correct[1, ordinary].mean() - correct[0, ordinary].mean())
    gate = {
        "top1_at_least_plus_0_30pp": bootstrap["delta"] >= 0.003,
        "group_ci95_lower_positive": bootstrap["ci95_low"] > 0.0,
        "macro_nonnegative": macro_delta >= 0.0,
        "tail_at_least_minus_0_25pp": tail_delta >= -0.0025,
        "ordinary_nonconflict_nonnegative": ordinary_delta >= 0.0,
    }
    result = {
        "scope": "P17 same-trajectory raw vs EMA; Stage2 train-derived fold2 only; no test",
        "raw_checkpoint_sha256": RAW_SHA,
        "ema_checkpoint_sha256": EMA_SHA,
        "raw_report_sha256": sha256_file(RAW / "report.json"),
        "ema_report_sha256": sha256_file(EMA / "report.json"),
        "raw_predictions_sha256": sha256_file(paths[0]),
        "ema_predictions_sha256": sha256_file(paths[1]),
        "manifest_sha256": sha256_file(manifest),
        "validation_count": 29737,
        "ordinary_count": int(ordinary.sum()),
        "top1": {"raw": float(correct[0].mean()), "ema": float(correct[1].mean()),
                 "delta": bootstrap["delta"]},
        "macro_delta": float(macro_delta),
        "tail_class_mean_recall_delta": float(tail_delta),
        "ordinary_nonconflict_delta": ordinary_delta,
        "paired_wins": int(((correct[0] == 0) & (correct[1] == 1)).sum()),
        "paired_losses": int(((correct[0] == 1) & (correct[1] == 0)).sum()),
        "near_duplicate_group_bootstrap": bootstrap,
        "gate": gate,
        "gate_passed": all(gate.values()),
        "caveat": "A passed fold2 alone would not authorize fold3, full training, or test.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(OUTPUT.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, OUTPUT)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
