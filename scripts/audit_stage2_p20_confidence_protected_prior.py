"""P20 fixed high-confidence protection audit on P7 train-derived logits only."""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from aictrack1.data.manifest import sha256_file  # noqa: E402
from aictrack1.engine.training_queue import TrainingLease  # noqa: E402
from aictrack1.evaluation.balanced_prior import fit_uniform_soft_bias  # noqa: E402
from aictrack1.evaluation.confidence_protected_prior import (  # noqa: E402
    confidence_protected_predictions,
)
from aictrack1.evaluation.feature_geometry import grouped_bootstrap_delta  # noqa: E402
from scripts.audit_stage2_balanced_prior import _balanced_proxy_indices  # noqa: E402

DIAGNOSTICS = ROOT / "artifacts" / "diagnostics"
REPORTS = ROOT / "artifacts" / "reports"
MANIFEST = ROOT / "artifacts" / "audits" / "stage2" / "train" / "samples.csv"
P7_REPORT = REPORTS / "20260928-p7-balanced-prior-fold2-fold3.json"
P7_REPORT_SHA = "8122c9c96fb80823db9d19edb403d8d147d26a32c43b7b48fe7cca9db311d703"
BUNDLES = (
    ("fold2", "20260928-p7-fold2-p5-sce-fourview-logits", "0aecb9b15eec86b781fa8d98c98d78ccb14eb636404fa7cfd86146cd9166c06d"),
    ("fold3", "20260928-p7-fold3-p5-sce-fourview-logits", "c0e9209f52bc7b7b44e2a322935c3452f2250bd34af4c8b0a2cc274f49319a1e"),
)
SEEDS = (20260928, 20260929, 20260930)


def _groups() -> dict[str, str]:
    result: dict[str, str] = {}
    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {
            "sample_id", "source_split", "near_duplicate_group"
        }.issubset(reader.fieldnames):
            raise ValueError("incomplete Stage2 train manifest")
        for row in reader:
            sample_id = row["sample_id"]
            if row["source_split"] != "train" or sample_id in result:
                raise ValueError("invalid or repeated Stage2 train sample")
            near_group = row["near_duplicate_group"]
            result[sample_id] = f"near:{near_group}" if near_group else f"singleton:{sample_id}"
    if len(result) != 148695:
        raise ValueError("incomplete Stage2 train population")
    return result


def _sample_ids(directory: Path, targets: torch.Tensor, groups: dict[str, str]) -> list[str]:
    path = directory / "paired_predictions.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"sample_id", "target"}.issubset(reader.fieldnames):
            raise ValueError("P7 paired prediction columns missing")
        rows = list(reader)
    ids = [row["sample_id"] for row in rows]
    if len(ids) != 29737 or len(set(ids)) != 29737 or any(item not in groups for item in ids):
        raise ValueError("P7 predictions are not one complete train-derived validation fold")
    if not torch.equal(torch.tensor([int(row["target"]) for row in rows]), targets):
        raise ValueError("P7 prediction targets differ from fused logits")
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--hard-seconds", type=float, default=1200.0)
    parser.add_argument(
        "--lock-path", type=Path,
        default=ROOT / "artifacts" / "queues" / "training.lock.json",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(REPORTS.resolve()) or output.exists():
        raise ValueError("output must be a new report inside artifacts/reports")
    if args.hard_seconds <= 0:
        raise ValueError("hard-seconds must be positive")
    lock_path = args.lock_path.resolve()
    if lock_path.name != "training.lock.json" or not lock_path.parent.is_dir():
        raise ValueError("lock-path must be an existing queue directory's training.lock.json")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if sha256_file(P7_REPORT) != P7_REPORT_SHA:
        raise ValueError("P7 frozen selection report SHA mismatch")
    selected = json.loads(P7_REPORT.read_text(encoding="utf-8"))
    if selected.get("selected_candidate") != "uniform_soft_t0.5_s0.75" or not selected.get("gate_passed"):
        raise ValueError("P7 frozen selection is not the expected baseline")
    groups = _groups()
    manifest_sha = sha256_file(MANIFEST)
    deadline = time.monotonic() + args.hard_seconds
    lease = TrainingLease.acquire(
        lock_path,
        owner="stage2-p20-train-derived-audit",
        config=str(P7_REPORT),
        output_dir=str(output),
        command=tuple(sys.argv),
    )
    atexit.register(lease.release)
    all_baseline: list[np.ndarray] = []
    all_candidate: list[np.ndarray] = []
    all_groups: list[str] = []
    fold_reports = []
    try:
        for fold, directory_name, expected_sha in BUNDLES:
            if time.monotonic() >= deadline:
                raise TimeoutError("P20 audit reached its pre-registered hard cutoff")
            directory = DIAGNOSTICS / directory_name
            bundle_path = directory / "fused_logits.pt"
            if sha256_file(bundle_path) != expected_sha:
                raise ValueError(f"{fold} fused logits SHA mismatch")
            source = json.loads((directory / "report.json").read_text(encoding="utf-8"))
            if (source.get("fused_logits", {}).get("sha256") != expected_sha
                    or source.get("smoke_only") is not False
                    or source.get("evaluated_count") != 29737
                    or source.get("four_view_resize") != 384
                    or source.get("manifest_sha256") != manifest_sha
                    or "train-derived" not in source.get("scope", "")):
                raise ValueError(f"{fold} is not the frozen full train-derived P7 source")
            payload = torch.load(bundle_path, map_location="cpu", weights_only=True)
            if (payload.get("schema") != "aictrack1.stage2_train_derived_fused_logits.v1"
                    or "train-derived" not in str(payload.get("scope", ""))
                    or any(payload.get(key) != source.get(key) for key in (
                        "checkpoint_sha256", "split_sha256", "manifest_sha256", "class_mapping_sha256"
                    ))):
                raise ValueError(f"{fold} bundle identity mismatch")
            logits = payload["logits"].float()
            targets = payload["targets"].long()
            if logits.shape != (29737, 750) or targets.shape != (29737,) or not torch.isfinite(logits).all():
                raise ValueError(f"{fold} bundle tensor shape or values invalid")
            ids = _sample_ids(directory, targets, groups)
            proxy_reports = []
            for seed in SEEDS:
                if time.monotonic() >= deadline:
                    raise TimeoutError("P20 audit reached its pre-registered hard cutoff")
                indices = _balanced_proxy_indices(targets, num_classes=750, samples_per_class=50, seed=seed)
                proxy = logits[indices].to(args.device)
                truth = targets[indices].to(args.device)
                bias, fit = fit_uniform_soft_bias(proxy, temperature_multiplier=0.5, iterations=60)
                p7, p20, protected = confidence_protected_predictions(
                    proxy, bias, median_margin=float(fit["median_top1_top2_margin"])
                )
                baseline = p7.eq(truth).cpu().numpy().astype(np.float64)
                candidate = p20.eq(truth).cpu().numpy().astype(np.float64)
                selected_groups = [groups[ids[index]] for index in indices.tolist()]
                bootstrap = grouped_bootstrap_delta(
                    baseline, candidate, selected_groups, seed=20260925, replicates=500
                )
                all_baseline.append(baseline)
                all_candidate.append(candidate)
                all_groups.extend(selected_groups)
                proxy_reports.append({
                    "seed": seed,
                    "p7_top1": float(baseline.mean()),
                    "p20_top1": float(candidate.mean()),
                    "delta_pp": 100 * bootstrap["delta"],
                    "ci95_pp": [100 * bootstrap["ci95_low"], 100 * bootstrap["ci95_high"]],
                    "protected_count": int(protected.sum().item()),
                    "changed_predictions": int(p7.ne(p20).sum().item()),
                    "p20_wins": int(((baseline == 0) & (candidate == 1)).sum()),
                    "p7_wins": int(((baseline == 1) & (candidate == 0)).sum()),
                    "fit": fit,
                })
            fold_reports.append({
                "fold": fold,
                "bundle_sha256": expected_sha,
                "checkpoint_sha256": source["checkpoint_sha256"],
                "proxy_results": proxy_reports,
                "mean_delta_pp": sum(item["delta_pp"] for item in proxy_reports) / len(proxy_reports),
            })
        combined = grouped_bootstrap_delta(
            np.concatenate(all_baseline), np.concatenate(all_candidate), all_groups,
            seed=20260925, replicates=500,
        )
        gates = {
            "both_folds_positive": all(item["mean_delta_pp"] > 0 for item in fold_reports),
            "overall_at_least_plus_0_15pp": 100 * combined["delta"] >= 0.15,
            "group_ci95_lower_positive": combined["ci95_low"] > 0,
        }
        result = {
            "schema": "aictrack1.stage2_p20_confidence_protected_prior_audit.v1",
            "scope": "Stage2 train-derived balanced proxies only; no test access",
            "baseline": "P7 uniform_soft_t0.5_s0.75",
            "candidate": "P7 with raw-margin >= 2 * P7 median-margin protection",
            "p7_report_sha256": P7_REPORT_SHA,
            "train_manifest_sha256": manifest_sha,
            "folds": fold_reports,
            "combined_delta_pp": 100 * combined["delta"],
            "combined_ci95_pp": [100 * combined["ci95_low"], 100 * combined["ci95_high"]],
            "combined_group_count": combined["group_count"],
            "gate": gates,
            "gate_passed": all(gates.values()),
            "caveat": "Fold overlap, resampled proxies and noisy labels limit generalization; not an online score estimate",
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    finally:
        lease.release()
        atexit.unregister(lease.release)
    print(json.dumps({"gate_passed": result["gate_passed"], "combined_delta_pp": result["combined_delta_pp"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
