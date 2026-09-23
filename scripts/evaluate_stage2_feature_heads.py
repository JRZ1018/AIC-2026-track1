"""Evaluate one or more Stage 2 heads on a provenance-bound train feature cache.

Only the fixed validation partition is scored. No competition test path is
accepted or read. Original-H1 and C2-feature-trained heads can be compared on
the exact same C2 features.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.class_mapping import ClassMapping  # noqa: E402
from aictrack1.data.contracts import load_dataset_config  # noqa: E402
from aictrack1.data.feature_cache import load_feature_cache  # noqa: E402
from aictrack1.data.manifest import read_manifest, sha256_file  # noqa: E402
from aictrack1.evaluation.classification import classification_diagnostics  # noqa: E402
from aictrack1.evaluation.long_tail import long_tail_diagnostics, stratify_classes  # noqa: E402
from aictrack1.engine.provenance import CheckpointProvenance  # noqa: E402
from aictrack1.models.heads import load_classifier_head_checkpoint  # noqa: E402


@torch.inference_mode()
def _predict(head: torch.nn.Module, features: torch.Tensor, device: torch.device) -> torch.Tensor:
    head = head.to(device).eval()
    predictions = []
    for chunk in features.split(4096):
        predictions.append(head(chunk.to(device).float()).argmax(dim=1).cpu())
    return torch.cat(predictions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--expected-peft-checkpoint-sha256", required=True)
    parser.add_argument("--head", required=True, action="append", help="LABEL=CHECKPOINT")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_dataset_config(args.dataset_config.resolve())
    if config.competition_stage != "stage2":
        raise ValueError("evaluation requires the Stage 2 train-only contract")
    mapping = ClassMapping.from_csv(config.class_mapping)
    cache = load_feature_cache(args.feature_cache.resolve())
    manifest_path, split_path = args.manifest.resolve(), args.split.resolve()
    if cache.metadata.source != "train" or cache.metadata.expected_record_count != config.expected_train_count:
        raise ValueError("a complete Stage 2 train-only cache is required")
    if cache.metadata.dataset_stage != config.stage:
        raise ValueError("feature cache dataset stage mismatch")
    expected = {
        "manifest_sha256": sha256_file(manifest_path),
        "split_sha256": sha256_file(split_path),
        "class_mapping_sha256": mapping.digest(),
        "peft_checkpoint_sha256": args.expected_peft_checkpoint_sha256,
    }
    for field, value in expected.items():
        if getattr(cache.metadata, field) != value:
            raise ValueError(f"feature cache {field} mismatch")
    records = read_manifest(manifest_path)
    if len(records) != len(cache.sample_ids) or any(record.source_split != "train" for record in records):
        raise ValueError("feature cache is not a complete Stage 2 train manifest")
    record_by_id = {record.sample_id: record for record in records}
    if set(record_by_id) != set(cache.sample_ids):
        raise ValueError("feature cache sample IDs differ from manifest")
    if any(
        int(cache.labels[index]) != record_by_id[sample_id].internal_class_index
        for index, sample_id in enumerate(cache.sample_ids)
    ):
        raise ValueError("feature cache labels differ from train manifest")
    hash_labels: dict[str, set[str]] = defaultdict(set)
    for record in records:
        hash_labels[record.sha256].add(record.official_class_id)
    conflict_ids = {
        record.sample_id for record in records if len(hash_labels[record.sha256]) > 1
    }
    train_labels = cache.labels[torch.tensor([part == "train" for part in cache.partitions])]
    class_counts = torch.bincount(train_labels, minlength=len(mapping))
    strata = stratify_classes(class_counts, head_fraction=0.2, mid_fraction=0.3, seed=config.split.seed)
    validation_positions = [index for index, part in enumerate(cache.partitions) if part == "validation"]
    if not validation_positions:
        raise ValueError("cache has no validation partition")
    positions = torch.tensor(validation_positions, dtype=torch.long)
    features, targets = cache.features.index_select(0, positions), cache.labels.index_select(0, positions)
    conflict_mask = torch.tensor([cache.sample_ids[index] in conflict_ids for index in validation_positions])
    if not conflict_mask.any() or bool(conflict_mask.all()):
        raise ValueError("conflict and non-conflict validation partitions must both be present")
    provenance = CheckpointProvenance(
        competition_stage="stage2",
        class_mapping_sha256=mapping.digest(),
        train_manifest_sha256=expected["manifest_sha256"],
        split_sha256=expected["split_sha256"],
        clip_weight_sha256=cache.metadata.clip_weight_sha256,
        num_classes=len(mapping),
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    result = {
        "scope": "Stage 2 train-derived fixed validation only; test not accessed",
        "feature_cache": str(args.feature_cache.resolve()),
        "peft_checkpoint_sha256": cache.metadata.peft_checkpoint_sha256,
        "split_sha256": expected["split_sha256"],
        "validation_count": len(validation_positions),
        "validation_exact_conflict_count": int(conflict_mask.sum()),
        "heads": {},
    }
    for specification in args.head:
        if "=" not in specification:
            raise ValueError("--head must be LABEL=CHECKPOINT")
        label, raw_path = specification.split("=", 1)
        if not label or label in result["heads"]:
            raise ValueError("head labels must be non-empty and unique")
        checkpoint_path = Path(raw_path).resolve()
        head, checkpoint = load_classifier_head_checkpoint(
            checkpoint_path,
            feature_dim=cache.metadata.feature_dim,
            num_classes=len(mapping),
            expected_clip_weight_sha256=cache.metadata.clip_weight_sha256,
            expected_class_mapping_sha256=mapping.digest(),
            expected_split_sha256=expected["split_sha256"],
            expected_provenance=provenance,
            forbid_full_train_initializer=True,
        )
        source_sha = checkpoint["feature_cache_metadata"].get("peft_checkpoint_sha256")
        if source_sha is not None and source_sha != cache.metadata.peft_checkpoint_sha256:
            raise ValueError(f"head {label} was trained on another PEFT checkpoint")
        predictions = _predict(head, features, device)
        correct = predictions.eq(targets)
        total = torch.bincount(targets, minlength=len(mapping))
        correct_by_class = torch.bincount(targets[correct], minlength=len(mapping))
        predicted_by_class = torch.bincount(predictions, minlength=len(mapping))
        result["heads"][label] = {
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "source_peft_checkpoint_sha256": source_sha,
            "all": classification_diagnostics(predictions, targets, num_classes=len(mapping)),
            "long_tail": long_tail_diagnostics(predictions, targets, num_classes=len(mapping), stratification=strata),
            "exact_conflict": classification_diagnostics(predictions[conflict_mask], targets[conflict_mask], num_classes=len(mapping)),
            "non_conflict": classification_diagnostics(predictions[~conflict_mask], targets[~conflict_mask], num_classes=len(mapping)),
            "per_class": [
                {"class_index": index, "validation_support": int(total[index]), "correct": int(correct_by_class[index]), "predicted_count": int(predicted_by_class[index])}
                for index in range(len(mapping))
            ],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({label: value["all"]["top1"] for label, value in result["heads"].items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
