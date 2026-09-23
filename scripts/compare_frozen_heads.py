"""Compare frozen-feature heads with class-balanced Stage2 diagnostics.

Only a current-stage train feature cache is accepted.  The script evaluates the
fixed validation partition and derives head/mid/tail strata from the fixed train
partition; it never reads competition test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.feature_cache import load_feature_cache, select_feature_partition  # noqa: E402
from aictrack1.engine.provenance import CheckpointProvenance, validate_checkpoint_provenance  # noqa: E402
from aictrack1.evaluation.long_tail import long_tail_diagnostics, stratify_classes  # noqa: E402
from aictrack1.models.heads import build_classifier_head  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def advancement_gate(
    candidate: dict,
    baseline: dict,
    *,
    max_top1_drop: float = 0.002,
) -> dict:
    """Apply the frozen Day-1 H1 gate using fixed-validation metrics."""

    candidate_validation = candidate["validation"]
    baseline_validation = baseline["validation"]
    deltas = {
        "top1": candidate_validation["top1"] - baseline_validation["top1"],
        "macro_accuracy": (
            candidate_validation["macro_accuracy"]
            - baseline_validation["macro_accuracy"]
        ),
        "tail_accuracy": (
            candidate_validation["strata"]["tail"]["accuracy"]
            - baseline_validation["strata"]["tail"]["accuracy"]
        ),
        "zero_recall_class_count": (
            candidate_validation["zero_recall_class_count"]
            - baseline_validation["zero_recall_class_count"]
        ),
    }
    checks = {
        "top1_not_down_more_than_limit": deltas["top1"] >= -max_top1_drop,
        "macro_improved": deltas["macro_accuracy"] > 0.0,
        "tail_improved": deltas["tail_accuracy"] > 0.0,
    }
    return {
        "advance": all(checks.values()),
        "checks": checks,
        "deltas": deltas,
        "deltas_percentage_points": {
            key: value * 100.0
            for key, value in deltas.items()
            if key != "zero_recall_class_count"
        },
        "max_allowed_top1_drop": max_top1_drop,
    }


@torch.inference_mode()
def _predictions(model, features: torch.Tensor, *, device: torch.device, batch_size: int):
    model.eval()
    outputs = []
    for (batch,) in DataLoader(TensorDataset(features), batch_size=batch_size):
        outputs.append(model(batch.to(device).float()).float().argmax(dim=1).cpu())
    return torch.cat(outputs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", required=True, help="LABEL=PATH")
    parser.add_argument("--baseline", help="checkpoint label used for advancement gates")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    cache = load_feature_cache(args.feature_cache.resolve())
    if cache.metadata.source != "train":
        raise ValueError("frozen-head comparison refuses test feature caches")
    train_features, train_labels = select_feature_partition(cache, "train")
    del train_features
    validation_features, validation_labels = select_feature_partition(cache, "validation")
    num_classes = int(train_labels.max()) + 1
    class_counts = torch.bincount(train_labels.long(), minlength=num_classes)
    if bool((class_counts == 0).any()):
        raise ValueError("fixed train partition must contain every class")
    strata = stratify_classes(
        class_counts,
        head_fraction=0.2,
        mid_fraction=0.3,
        seed=20260921,
    )
    expected = CheckpointProvenance(
        competition_stage="stage2",
        class_mapping_sha256=str(cache.metadata.class_mapping_sha256),
        train_manifest_sha256=str(cache.metadata.manifest_sha256),
        split_sha256=str(cache.metadata.split_sha256),
        clip_weight_sha256=str(cache.metadata.clip_weight_sha256),
        num_classes=num_classes,
    )
    device = torch.device(args.device)
    reports = {}
    for specification in args.checkpoint:
        if "=" not in specification:
            raise ValueError("--checkpoint must use LABEL=PATH")
        label, raw_path = specification.split("=", 1)
        path = Path(raw_path).resolve()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        validate_checkpoint_provenance(
            checkpoint,
            expected,
            context=f"frozen-head checkpoint {label}",
        )
        head_raw = checkpoint["head"]
        model = build_classifier_head(
            str(head_raw["kind"]),
            cache.metadata.feature_dim,
            num_classes,
            cosine_initial_scale=float(head_raw.get("initial_scale", 10.0)),
            cosine_learnable_scale=bool(head_raw.get("learnable_scale", True)),
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        predictions = _predictions(
            model,
            validation_features,
            device=device,
            batch_size=args.batch_size,
        )
        reports[label] = {
            "checkpoint": str(path),
            "checkpoint_sha256": _sha256(path),
            "selection_rule": checkpoint.get("selection_rule"),
            "class_weighting": checkpoint.get("class_weighting", {"mode": "legacy_natural"}),
            "validation": long_tail_diagnostics(
                predictions,
                validation_labels,
                num_classes=num_classes,
                stratification=strata,
            ),
        }
    if args.baseline is not None and args.baseline not in reports:
        raise ValueError("--baseline must name one of the checkpoint labels")
    gates = {}
    selected_candidate = None
    if args.baseline is not None:
        gates = {
            label: advancement_gate(report, reports[args.baseline])
            for label, report in reports.items()
            if label != args.baseline
        }
        eligible = [label for label, gate in gates.items() if gate["advance"]]
        if eligible:
            selected_candidate = max(
                eligible,
                key=lambda label: reports[label]["validation"]["top1"],
            )
    payload = {
        "schema": "aictrack1.frozen_head_comparison.v2",
        "scope": "Stage2 fixed validation only; test data not accessed",
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "feature_cache": str(args.feature_cache.resolve()),
        "feature_cache_manifest_sha256": cache.metadata.manifest_sha256,
        "feature_cache_split_sha256": cache.metadata.split_sha256,
        "stratification": {
            "head_classes": strata.head_count,
            "mid_classes": strata.mid_count,
            "tail_classes": num_classes - strata.head_count - strata.mid_count,
            "train_support_min": int(class_counts.min()),
            "train_support_median": float(class_counts.float().median()),
            "train_support_max": int(class_counts.max()),
        },
        "heads": reports,
        "advancement": {
            "baseline": args.baseline,
            "gate": (
                "macro and tail must improve; Top-1 may not drop by more than 0.2 percentage points"
            ),
            "candidates": gates,
            "selected_candidate": selected_candidate,
        },
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
