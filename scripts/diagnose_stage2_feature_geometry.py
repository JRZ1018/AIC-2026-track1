"""Compare original OpenAI CLIP and fixed-fold C2 train-only feature geometry.

This diagnostic never reads competition test data or fits on validation rows.
The common validation labels are noisy observed labels, not clean ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.class_mapping import ClassMapping  # noqa: E402
from aictrack1.data.contracts import load_dataset_config  # noqa: E402
from aictrack1.data.feature_cache import LoadedFeatureCache, load_feature_cache  # noqa: E402
from aictrack1.data.manifest import read_manifest, sha256_file  # noqa: E402
from aictrack1.evaluation.feature_geometry import (  # noqa: E402
    centroid_validation_scores,
    class_centroids,
    grouped_bootstrap_delta,
    nearest_neighbor_label_agreement,
)
from aictrack1.evaluation.long_tail import stratify_classes  # noqa: E402


def _validate_cache_pair(
    original: LoadedFeatureCache,
    c2: LoadedFeatureCache,
    *,
    expected_count: int,
    manifest_sha: str,
    split_sha: str,
    mapping_sha: str,
    peft_sha: str,
) -> None:
    for cache in (original, c2):
        metadata = cache.metadata
        if (
            metadata.source != "train"
            or metadata.expected_record_count != expected_count
            or metadata.manifest_sha256 != manifest_sha
            or metadata.split_sha256 != split_sha
            or metadata.class_mapping_sha256 != mapping_sha
            or not metadata.normalized
        ):
            raise ValueError("feature cache does not match the Stage 2 train fold")
        if not torch.isfinite(cache.features).all() or (cache.labels < 0).any():
            raise ValueError("feature cache contains non-finite features or missing labels")
    if original.metadata.peft_checkpoint_sha256 is not None:
        raise ValueError("original CLIP cache must not contain PEFT features")
    if c2.metadata.peft_checkpoint_sha256 != peft_sha:
        raise ValueError("C2 cache checkpoint SHA-256 mismatch")
    for field in ("dataset_stage", "clip_weight_sha256", "preprocessing", "feature_dim"):
        if getattr(original.metadata, field) != getattr(c2.metadata, field):
            raise ValueError(f"feature cache {field} mismatch")
    if (
        original.sample_ids != c2.sample_ids
        or original.partitions != c2.partitions
        or not torch.equal(original.labels, c2.labels)
    ):
        raise ValueError("feature caches do not contain the same ordered samples and labels")


def _mean(values: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    chosen = values if mask is None else values[mask]
    return float(chosen.float().mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--original-cache", type=Path, required=True)
    parser.add_argument("--c2-cache", type=Path, required=True)
    parser.add_argument("--expected-peft-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--train-knn-count", type=int, default=12000)
    parser.add_argument("--validation-knn-count", type=int, default=3000)
    parser.add_argument("--sample-seed", type=int, default=20260923)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    args = parser.parse_args()
    if args.train_knn_count < 5 or args.validation_knn_count < 1:
        raise ValueError("kNN samples must include at least five train rows and one validation row")
    config = load_dataset_config(args.dataset_config.resolve())
    if config.competition_stage != "stage2":
        raise ValueError("feature geometry requires the Stage 2 train contract")
    mapping = ClassMapping.from_csv(config.class_mapping)
    manifest_path, split_path = args.manifest.resolve(), args.split.resolve()
    manifest_sha, split_sha = sha256_file(manifest_path), sha256_file(split_path)
    started = time.perf_counter()
    original = load_feature_cache(args.original_cache.resolve())
    c2 = load_feature_cache(args.c2_cache.resolve())
    _validate_cache_pair(
        original,
        c2,
        expected_count=config.expected_train_count,
        manifest_sha=manifest_sha,
        split_sha=split_sha,
        mapping_sha=mapping.digest(),
        peft_sha=args.expected_peft_checkpoint_sha256,
    )
    records = read_manifest(manifest_path)
    record_by_id = {record.sample_id: record for record in records}
    if len(record_by_id) != len(original.sample_ids) or set(record_by_id) != set(original.sample_ids):
        raise ValueError("manifest sample IDs differ from complete feature caches")
    hash_labels: dict[str, set[str]] = {}
    for record in records:
        hash_labels.setdefault(record.sha256, set()).add(record.official_class_id)
    train_positions = torch.tensor(
        [index for index, partition in enumerate(original.partitions) if partition == "train"]
    )
    validation_positions = torch.tensor(
        [index for index, partition in enumerate(original.partitions) if partition == "validation"]
    )
    if len(train_positions) + len(validation_positions) != len(original.sample_ids):
        raise ValueError("cache contains unexpected partitions")
    if args.train_knn_count > len(train_positions) or args.validation_knn_count > len(validation_positions):
        raise ValueError("requested kNN sample count exceeds the fixed fold")
    validation_ids = [original.sample_ids[index] for index in validation_positions.tolist()]
    validation_groups = [
        f"near:{record_by_id[sample_id].near_duplicate_group}"
        if record_by_id[sample_id].near_duplicate_group
        else f"singleton:{sample_id}"
        for sample_id in validation_ids
    ]
    validation_conflicts = torch.tensor(
        [len(hash_labels[record_by_id[sample_id].sha256]) > 1 for sample_id in validation_ids],
        dtype=torch.bool,
    )
    generator = torch.Generator().manual_seed(args.sample_seed)
    train_knn_positions = train_positions[
        torch.randperm(len(train_positions), generator=generator)[: args.train_knn_count]
    ]
    validation_knn_local = torch.randperm(len(validation_positions), generator=generator)[
        : args.validation_knn_count
    ]
    validation_knn_positions = validation_positions[validation_knn_local]
    validation_knn_groups = [validation_groups[index] for index in validation_knn_local.tolist()]
    validation_knn_conflicts = validation_conflicts[validation_knn_local]
    train_counts = torch.bincount(
        original.labels.index_select(0, train_positions), minlength=len(mapping)
    )
    strata = stratify_classes(
        train_counts, head_fraction=0.2, mid_fraction=0.3, seed=config.split.seed
    )
    class_strata = strata.assignments
    validation_strata = [class_strata[int(original.labels[index])] for index in validation_positions]
    validation_knn_strata = [validation_strata[index] for index in validation_knn_local.tolist()]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    per_model: dict[str, dict] = {}
    paired: dict[str, dict[str, np.ndarray]] = {}
    for name, cache in (("original_clip", original), ("c2_fixed", c2)):
        train_features = cache.features.index_select(0, train_positions)
        train_labels = cache.labels.index_select(0, train_positions)
        centroids = class_centroids(train_features, train_labels, num_classes=len(mapping))
        val_features = cache.features.index_select(0, validation_positions)
        val_labels = cache.labels.index_select(0, validation_positions)
        predictions, true_similarity, other_similarity, margins = centroid_validation_scores(
            val_features, val_labels, centroids, device=device
        )
        correct = predictions.eq(val_labels)
        knn_top1, knn_top5_fraction = nearest_neighbor_label_agreement(
            cache.features.index_select(0, train_knn_positions),
            cache.labels.index_select(0, train_knn_positions),
            cache.features.index_select(0, validation_knn_positions),
            cache.labels.index_select(0, validation_knn_positions),
            device=device,
        )
        centroid_similarity = centroids @ centroids.T
        centroid_similarity.fill_diagonal_(-torch.inf)
        nearest_other_centroid = centroid_similarity.max(dim=1).values
        by_stratum = {}
        for stratum in ("head", "mid", "tail"):
            mask = torch.tensor([value == stratum for value in validation_strata])
            knn_mask = torch.tensor([value == stratum for value in validation_knn_strata])
            by_stratum[stratum] = {
                "validation_count": int(mask.sum()),
                "centroid_observed_label_top1": _mean(correct, mask),
                "centroid_margin_mean": _mean(margins, mask),
                "knn_validation_count": int(knn_mask.sum()),
                "knn_top1_label_agreement": _mean(knn_top1, knn_mask),
            }
        per_model[name] = {
            "checkpoint_sha256": cache.metadata.peft_checkpoint_sha256,
            "centroid": {
                "observed_label_top1": _mean(correct),
                "true_class_cosine_mean": _mean(true_similarity),
                "nearest_wrong_class_cosine_mean": _mean(other_similarity),
                "margin_mean": _mean(margins),
                "margin_median": float(margins.median()),
                "positive_margin_fraction": _mean(margins > 0),
                "exact_conflict_observed_label_top1": _mean(correct, validation_conflicts),
                "non_conflict_observed_label_top1": _mean(correct, ~validation_conflicts),
                "nearest_other_train_centroid_cosine_mean": _mean(nearest_other_centroid),
            },
            "knn": {
                "top1_observed_label_agreement": _mean(knn_top1),
                "top5_neighbor_label_fraction": _mean(knn_top5_fraction),
                "exact_conflict_top1_agreement": _mean(knn_top1, validation_knn_conflicts),
                "non_conflict_top1_agreement": _mean(knn_top1, ~validation_knn_conflicts),
            },
            "by_stratum": by_stratum,
        }
        paired[name] = {
            "centroid_correct": correct.numpy().astype(np.float64),
            "centroid_margin": margins.numpy().astype(np.float64),
            "knn_top1": knn_top1.numpy().astype(np.float64),
        }
    confidence = {
        "centroid_top1_delta": grouped_bootstrap_delta(
            paired["original_clip"]["centroid_correct"],
            paired["c2_fixed"]["centroid_correct"],
            validation_groups,
            seed=args.sample_seed,
            replicates=args.bootstrap_replicates,
        ),
        "centroid_margin_delta": grouped_bootstrap_delta(
            paired["original_clip"]["centroid_margin"],
            paired["c2_fixed"]["centroid_margin"],
            validation_groups,
            seed=args.sample_seed + 1,
            replicates=args.bootstrap_replicates,
        ),
        "knn_top1_agreement_delta": grouped_bootstrap_delta(
            paired["original_clip"]["knn_top1"],
            paired["c2_fixed"]["knn_top1"],
            validation_knn_groups,
            seed=args.sample_seed + 2,
            replicates=args.bootstrap_replicates,
        ),
    }
    result = {
        "scope": "Stage 2 train-derived fixed validation only; test not accessed",
        "manifest_sha256": manifest_sha,
        "split_sha256": split_sha,
        "class_mapping_sha256": mapping.digest(),
        "clip_weight_sha256": original.metadata.clip_weight_sha256,
        "sample_seed": args.sample_seed,
        "train_count": len(train_positions),
        "validation_count": len(validation_positions),
        "validation_exact_conflict_count": int(validation_conflicts.sum()),
        "knn_train_sample_count": args.train_knn_count,
        "knn_validation_sample_count": args.validation_knn_count,
        "knn_validation_exact_conflict_count": int(validation_knn_conflicts.sum()),
        "models": per_model,
        "paired_group_bootstrap": confidence,
        "bootstrap_caveat": "Conditional on fixed learned features and noisy observed labels; does not capture retraining variance or hidden test shift",
        "duration_seconds": time.perf_counter() - started,
        "peak_vram_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"duration_seconds": result["duration_seconds"], "models": per_model, "paired_group_bootstrap": confidence}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
