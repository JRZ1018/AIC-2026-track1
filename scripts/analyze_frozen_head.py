"""Analyze one frozen-head checkpoint without accessing competition test data."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.feature_cache import load_feature_cache
from aictrack1.evaluation.classification import calibration_diagnostics, classification_diagnostics
from aictrack1.models.heads import build_classifier_head


def _cross_label_conflict_ids(manifest: Path, conflicts: Path) -> set[str]:
    with conflicts.open("r", encoding="utf-8-sig", newline="") as handle:
        conflict_hashes = {str(row["sha256"]) for row in csv.DictReader(handle)}
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            str(row["sample_id"])
            for row in csv.DictReader(handle)
            if str(row["sha256"]) in conflict_hashes
        }


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--label-conflicts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--include-train", action="store_true")
    args = parser.parse_args()

    cache = load_feature_cache(args.cache)
    if cache.metadata.source != "train":
        raise ValueError("diagnostics refuse test feature caches")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["feature_cache_metadata"] != cache.metadata.__dict__:
        raise ValueError("checkpoint and feature cache metadata do not match")
    head_config = checkpoint["head"]
    loss_config = checkpoint.get("loss", {"kind": "cross_entropy"})
    num_classes = int(cache.labels.max()) + 1
    model = build_classifier_head(
        str(head_config["kind"]),
        cache.metadata.feature_dim,
        num_classes,
        cosine_initial_scale=float(head_config.get("initial_scale", 10.0)),
        cosine_learnable_scale=bool(head_config.get("learnable_scale", True)),
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model.to(device).eval()

    conflict_ids = _cross_label_conflict_ids(args.manifest, args.label_conflicts)

    def analyze_partition(partition: str) -> dict[str, object]:
        selection = torch.tensor(
            [value == partition for value in cache.partitions], dtype=torch.bool
        )
        features = cache.features[selection].float()
        targets = cache.labels[selection]
        sample_ids = [
            sample_id
            for sample_id, keep in zip(cache.sample_ids, selection.tolist(), strict=True)
            if keep
        ]
        logits = []
        for (batch,) in DataLoader(TensorDataset(features), batch_size=args.batch_size):
            logits.append(model(batch.to(device)).float().cpu())
        predictions = torch.cat(logits).argmax(dim=1)
        all_logits = torch.cat(logits)
        conflict_mask = torch.tensor([sample_id in conflict_ids for sample_id in sample_ids])
        result = {
            "all": classification_diagnostics(predictions, targets, num_classes=num_classes),
            "calibration": calibration_diagnostics(all_logits, targets),
            "cross_label_exact_conflicts": (
                classification_diagnostics(
                    predictions[conflict_mask], targets[conflict_mask], num_classes=num_classes
                )
                if conflict_mask.any()
                else None
            ),
            "remaining": classification_diagnostics(
                predictions[~conflict_mask], targets[~conflict_mask], num_classes=num_classes
            ),
            "conflict_sample_count": int(conflict_mask.sum()),
        }
        if loss_config.get("kind") == "logit_clip_cross_entropy":
            tau = float(loss_config["logit_clip_tau"])
            norms = torch.linalg.vector_norm(all_logits, ord=float(loss_config["logit_clip_p_norm"]), dim=1)
            result["logit_clip"] = {
                "tau": tau,
                "p_norm": float(loss_config["logit_clip_p_norm"]),
                "would_clip_fraction": float((norms > tau).float().mean()),
            }
        return result

    result = {
        "scope": "stage1 noisy train/validation only; competition test data not accessed",
        "loss": loss_config,
        "validation": analyze_partition("validation"),
    }
    if args.include_train:
        result["train"] = analyze_partition("train")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
