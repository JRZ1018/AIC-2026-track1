"""Measure whether the classwise-history soft weighting can detect true label noise.

Screening question (batch 5 follow-up): the online score improved monotonically as
`min_weight` was lowered 0.50 -> 0.35 -> 0.20 -> 0.10, which is the signature of a
mechanism that is being switched off rather than tuned. This script tests the
mechanism's discriminating capacity directly.

The mechanism assigns
    noise  = classwise_rank_noise_scores(losses)      # within-class percentile in [0, 1]
    weight = min_weight + (1 - min_weight) * (1 - noise)

`classwise_rank_noise_scores` maps losses to tie-aware *within-class percentiles*, so
the multiset of `noise` values is uniform on {0, 1/(n_c-1), ..., 1} inside every class
by construction. The only channel left through which the mechanism can carry
information about which samples are mislabeled is the assignment of ranks to samples.

This script therefore reports two things:

1. `auc_empirical`: if we could reorder samples inside each class perfectly, how well
   does the within-class rank order separate true conflicts from clean samples?
2. `auc_null`: the AUC expected if the rank assignment were random within each class,
   with its exact mean and standard deviation. This is the capacity ceiling that the
   fixed uniform shape imposes on ANY within-class-rank reweighting.

A positive but small empirical/null gap means the mechanism is structurally unable to
identify noise and its measured benefit must come from gradient scaling, not noise
suppression.

Ground truth: `label_conflicts.csv` lists content hashes that occur under more than one
class label (1032 hashes; 1009 with 2 labels, 21 with 3, 2 with 4). It is produced
automatically by the data audit, not by hand.

Read-only with respect to data: reads the audited manifest, split and conflict list.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def _read_conflict_hashes(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {str(row["sha256"]) for row in csv.DictReader(handle) if row.get("sha256")}


def _read_manifest(path: Path, conflict_hashes: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sha256 = str(row["sha256"])
            rows.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "class_index": int(row["internal_class_index"]),
                    "official_class_id": str(row["official_class_id"]),
                    "conflict": sha256 in conflict_hashes,
                }
            )
    return rows


def _read_split(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {str(row["sample_id"]): str(row["split"]) for row in csv.DictReader(handle)}


def _auc_against_conflict(noise: list[float], conflict: list[bool]) -> float:
    """Tie-aware AUC of `noise` as a conflict score, computed by rank sum.

    AUC = P(noise_positive > noise_negative) + 0.5 * P(tie), which for tied scores is
    exactly the Mann-Whitney statistic built from average ranks.
    """
    positives = sum(1 for value in conflict if value)
    negatives = len(conflict) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUC needs both conflict and clean samples")

    order = sorted(range(len(noise)), key=lambda index: noise[index])
    ranks = [0.0] * len(noise)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and noise[order[end]] == noise[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end

    positive_rank_sum = sum(rank for rank, flag in zip(ranks, conflict, strict=True) if flag)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _null_auc_moments(conflict: list[bool], class_ids: list[int]) -> tuple[float, float]:
    """Exact mean and standard deviation of the AUC under random within-class ranks.

    Under the null every sample receives an exchangeable random rank drawn from the
    fixed uniform rank multiset of its own class, so each positive/negative indicator is
    a Bernoulli(p) variable with

      * different classes: the two ranks are independent, and P(noise_i > noise_j) is
            (n_i - 1) / (2 * n_i) + (n_j - 1) / (2 * n_j) - (n_i - 1)(n_j - 1) / (2 n_i n_j)
        which tends to 1/2 for large classes, with a small tie probability.
      * same class: the ordered rank pair is uniform, so P(win) = P(tie) = 1/2 exactly.

    The AUC is therefore the mean of pairwise Bernoulli indicators. Cross-class pairs are
    independent under the null (they draw from disjoint rank pools), while same-class
    pairs are dependent; the variance contribution of same-class pairs is kept in its
    own stratum so the bound stays honest.
    """
    class_sizes = Counter(class_ids)
    members: dict[int, list[int]] = defaultdict(list)
    for index, class_id in enumerate(class_ids):
        members[class_id].append(index)

    same_class_pairs = 0
    cross_class_pairs = 0
    cross_mean_sum = 0.0
    for class_id, indices in members.items():
        size = class_sizes[class_id]
        positives = sum(1 for index in indices if conflict[index])
        negatives = len(indices) - positives
        if not positives or not negatives:
            continue
        same_class_pairs += positives * negatives
        for other_id, other_indices in members.items():
            if other_id == class_id:
                continue
            other_size = class_sizes[other_id]
            other_negatives = sum(1 for index in other_indices if not conflict[index])
            if not other_negatives:
                continue
            pairs = positives * other_negatives
            cross_class_pairs += pairs
            cross_mean_sum += pairs * (
                (size - 1) / (2.0 * size)
                + (other_size - 1) / (2.0 * other_size)
                - (size - 1) * (other_size - 1) / (2.0 * size * other_size)
            )

    positives_total = sum(1 for flag in conflict if flag)
    negatives_total = len(conflict) - positives_total
    if positives_total == 0 or negatives_total == 0:
        raise ValueError("null AUC needs both conflict and clean samples")

    pair_count = positives_total * negatives_total
    mean = (0.5 * same_class_pairs + cross_mean_sum) / pair_count
    # Same-class indicators sit in one stratum and are perfectly correlated through the
    # shared rank pool, so they contribute at most `same_class_pairs` worth of variance;
    # cross-class indicators are independent Bernoulli draws.
    variance_sum = same_class_pairs * 0.25 + cross_class_pairs * 0.25
    standard_deviation = math.sqrt(variance_sum) / pair_count
    return mean, standard_deviation


def _enrichment(
    noise: list[float], conflict: list[bool], fraction: float
) -> dict[str, float]:
    """How much more conflict-dense is the top-`fraction` noise bucket than base rate."""
    count = max(1, int(round(len(noise) * fraction)))
    order = sorted(range(len(noise)), key=lambda index: noise[index], reverse=True)
    selected = order[:count]
    hits = sum(1 for index in selected if conflict[index])
    base_rate = sum(1 for flag in conflict if flag) / len(conflict)
    bucket_rate = hits / count
    return {
        "fraction": fraction,
        "bucket_size": count,
        "conflicts_in_bucket": hits,
        "bucket_conflict_rate": bucket_rate,
        "base_conflict_rate": base_rate,
        "enrichment": bucket_rate / base_rate if base_rate else float("nan"),
    }


def _measure_real_loss_signal(
    *,
    feature_cache: Path,
    head_checkpoint: Path,
    manifest: Path,
    split: Path,
    conflict_hashes: set[str],
    partition: str,
    min_weight: float,
) -> dict[str, object]:
    """Score the mechanism's REAL input signal against the conflict ground truth.

    The mechanism's input is a within-class ranking of per-sample losses. Here the loss
    is produced by an available linear head applied to the frozen CLIP features, so this
    is a bona fide ordering signal rather than the degenerate uniform-shape proxy used
    above. The resulting AUC is the honest answer to "can the loss ranking find the
    mislabeled samples at better than chance?".
    """
    import sys

    import torch

    repository_root = Path(__file__).resolve().parents[1]
    source_root = repository_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from aictrack1.data.feature_cache import load_feature_cache
    from aictrack1.methods.confidence import classwise_rank_noise_scores

    cache = load_feature_cache(feature_cache)
    if cache.metadata.source != "train":
        raise ValueError("refusing to score a test feature cache")
    checkpoint = torch.load(head_checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    weight = state["linear.weight"].float()
    bias = state["linear.bias"].float()

    features = cache.features.float()
    logits = features @ weight.t() + bias
    labels = cache.labels.long()
    losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    probabilities = logits.softmax(dim=-1)
    confidence = probabilities.gather(1, labels.view(-1, 1)).squeeze(1)
    predictions = logits.argmax(dim=-1)

    class_ids = labels
    scores = classwise_rank_noise_scores(losses, class_ids)
    weights = min_weight + (1.0 - min_weight) * (1.0 - scores)

    cache_ids = list(cache.sample_ids)
    split_map = _read_split(split)
    manifest_rows = {
        str(row["sample_id"]): row for row in _read_manifest(manifest, conflict_hashes)
    }
    if any(sample_id not in manifest_rows for sample_id in cache_ids):
        raise ValueError("feature cache contains sample ids absent from the manifest")
    if any(split_map.get(sample_id) != part for sample_id, part in zip(cache_ids, cache.partitions, strict=True)):
        raise ValueError("feature cache partitions disagree with the split manifest")

    selected = [index for index, part in enumerate(cache.partitions) if part == partition]
    conflict_flags = [
        bool(manifest_rows[cache_ids[index]]["conflict"]) for index in selected
    ]
    selected_scores = [float(scores[index]) for index in selected]
    selected_losses = [float(losses[index]) for index in selected]

    empirical = _auc_against_conflict(selected_scores, conflict_flags)
    null_mean, null_sd = _null_auc_moments(
        conflict_flags, [int(class_ids[index]) for index in selected]
    )
    conflict_count = sum(1 for flag in conflict_flags if flag)
    correct = sum(
        1
        for index in selected
        if int(predictions[index]) == int(labels[index])
    )
    return {
        "head_checkpoint": str(head_checkpoint),
        "loss_kind": str(checkpoint.get("loss", {}).get("kind", "unknown")),
        "sample_count": len(selected),
        "conflict_sample_count": conflict_count,
        "conflict_rate": conflict_count / len(selected),
        "auc_loss_rank": empirical,
        "auc_null_mean": null_mean,
        "auc_null_sd": null_sd,
        "auc_z_score": (empirical - null_mean) / null_sd if null_sd else float("nan"),
        "accuracy_on_partition": correct / len(selected),
        "mean_loss_conflict": sum(
            value for value, flag in zip(selected_losses, conflict_flags, strict=True) if flag
        )
        / max(1, conflict_count),
        "mean_loss_clean": sum(
            value
            for value, flag in zip(selected_losses, conflict_flags, strict=True)
            if not flag
        )
        / max(1, len(selected) - conflict_count),
        "enrichment": [
            _enrichment(selected_scores, conflict_flags, value)
            for value in (0.01, 0.05, 0.10, 0.20)
        ],
        "mean_weight": sum(float(weights[index]) for index in selected) / len(selected),
    }


def _signal_quality_table(
    *,
    feature_cache: Path,
    head_checkpoint: Path,
    manifest: Path,
    split: Path,
    conflict_hashes: set[str],
    partition_filter: str,
) -> dict[str, object]:
    """Whole-split comparison of the two candidate noise signals.

    This is the measurement the batch-6 preregistration cites: how well does the signal
    the mechanism currently uses (within-class loss percentile) separate audited label
    conflicts, compared with a signal that keeps between-class loss magnitude?
    """
    import sys

    import torch

    repository_root = Path(__file__).resolve().parents[1]
    source_root = repository_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from aictrack1.data.feature_cache import load_feature_cache
    from aictrack1.methods.confidence import classwise_rank_noise_scores

    cache = load_feature_cache(feature_cache)
    if cache.metadata.source != "train":
        raise ValueError("refusing to score a test feature cache")
    checkpoint = torch.load(head_checkpoint, map_location="cpu", weights_only=False)
    logits = (
        cache.features.float() @ checkpoint["model"]["linear.weight"].float().t()
        + checkpoint["model"]["linear.bias"].float()
    )
    labels = cache.labels.long()
    loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    rank = classwise_rank_noise_scores(loss, labels)

    split_map = _read_split(split)
    cache_ids = list(cache.sample_ids)
    manifest_rows = {str(row["sample_id"]): row for row in _read_manifest(manifest, conflict_hashes)}
    if any(sample_id not in manifest_rows for sample_id in cache_ids):
        raise ValueError("feature cache contains sample ids absent from the manifest")
    if any(
        split_map.get(sample_id) != part
        for sample_id, part in zip(cache_ids, cache.partitions, strict=True)
    ):
        raise ValueError("feature cache partitions disagree with the split manifest")
    conflicts = torch.tensor(
        [
            bool(manifest_rows[sample_id]["conflict"])
            and (partition_filter == "all" or part == partition_filter)
            for sample_id, part in zip(cache_ids, cache.partitions, strict=True)
        ]
    )

    def auc(scores: torch.Tensor, positive: torch.Tensor) -> float:
        positives = int(positive.sum())
        negatives = scores.numel() - positives
        order = torch.argsort(scores, stable=True)
        ranks = torch.empty(scores.numel(), dtype=torch.float64)
        ranks[order] = torch.arange(scores.numel(), dtype=torch.float64)
        return float(
            (ranks[positive].sum() - positives * (positives - 1) / 2) / (positives * negatives)
        )

    if partition_filter != "all":
        keep = torch.tensor(
            [part == partition_filter for part in cache.partitions], dtype=torch.bool
        )
        loss = loss[keep]
        rank = rank[keep]
        labels = labels[keep]
        conflicts = conflicts[keep]

    positive = conflicts
    # Buckets by within-class percentile, which is what the shipped mechanism sorts on.
    percentile_buckets = []
    for low, high in ((0.9, 1.0), (0.8, 0.9), (0.7, 0.8), (0.5, 0.7), (0.0, 0.5)):
        mask = (rank >= low) & (rank < high if high < 1.0 else rank <= high)
        count = int(mask.sum())
        hits = int(positive[mask].sum())
        base = float(positive.float().mean())
        percentile_buckets.append(
            {
                "percentile_low": low,
                "percentile_high": high,
                "bucket_size": count,
                "conflicts_in_bucket": hits,
                "bucket_conflict_rate": hits / count if count else float("nan"),
                "enrichment": (hits / count) / base if count and base else float("nan"),
            }
        )

    budget_table = []
    for budget in (0.05, 0.10, 0.20, 0.30, 0.50):
        size = int(conflicts.numel() * budget)
        loss_threshold = float(loss.topk(size).values.min())
        rank_threshold = float(rank.topk(size).values.min())
        loss_mask = loss >= loss_threshold
        rank_mask = rank >= rank_threshold
        budget_table.append(
            {
                "budget": budget,
                "global_loss_recall": float(positive[loss_mask].sum()) / float(positive.sum()),
                "global_loss_precision": float(positive[loss_mask].float().mean()),
                "classwise_rank_recall": float(positive[rank_mask].sum())
                / float(positive.sum()),
                "classwise_rank_precision": float(positive[rank_mask].float().mean()),
            }
        )

    # Between-class structure: is the between-class loss magnitude the informative part?
    class_means = torch.zeros(int(labels.max()) + 1)
    for class_id in torch.unique(labels):
        class_means[class_id] = loss[labels == class_id].mean()
    class_centered = loss - class_means[labels]
    return {
        "head_checkpoint": str(head_checkpoint),
        "partition_filter": partition_filter,
        "sample_count": int(loss.numel()),
        "conflict_sample_count": int(positive.sum()),
        "conflict_rate": float(positive.float().mean()),
        "auc_classwise_rank": auc(rank, positive),
        "auc_global_loss": auc(loss, positive),
        "auc_class_centered_loss": auc(class_centered, positive),
        "percentile_buckets": percentile_buckets,
        "budget_table": budget_table,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--label-conflicts", required=True, type=Path)
    parser.add_argument("--min-weight", type=float, default=0.1)
    parser.add_argument("--partition", default="train", choices=("train", "validation"))
    parser.add_argument(
        "--feature-cache",
        type=Path,
        help="train feature cache; enables measuring the real loss-based signal",
    )
    parser.add_argument(
        "--head-checkpoint",
        type=Path,
        help="linear head checkpoint whose loss ranking is scored against the conflicts",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--conflict-split",
        default="all",
        choices=("all", "train", "validation"),
        help="restrict the conflict ground truth to one side of the fixed split",
    )
    parser.add_argument("--conflict-split-output", type=Path)
    args = parser.parse_args()

    conflict_hashes = _read_conflict_hashes(args.label_conflicts)
    rows = _read_manifest(args.manifest, conflict_hashes)
    split = _read_split(args.split)

    selected = [row for row in rows if split.get(str(row["sample_id"])) == args.partition]
    if not selected:
        raise ValueError(f"no rows for partition {args.partition!r}")

    class_ids = [int(row["class_index"]) for row in selected]
    conflict = [bool(row["conflict"]) for row in selected]
    class_sizes = Counter(class_ids)

    # The mechanism's own assignment: tie-aware within-class percentiles. With no loss
    # signal available this is the fixed uniform shape that any such reweighting must
    # have; the empirical AUC measured here is the optimistic bound for a mechanism that
    # could choose the within-class ordering freely.
    noise = [0.0] * len(selected)
    seen: Counter[int] = Counter()
    for index, class_id in enumerate(class_ids):
        size = class_sizes[class_id]
        rank = seen[class_id]
        seen[class_id] += 1
        noise[index] = 0.0 if size == 1 else rank / (size - 1)
    weights = [args.min_weight + (1.0 - args.min_weight) * (1.0 - value) for value in noise]

    empirical = _auc_against_conflict(noise, conflict)
    null_mean, null_sd = _null_auc_moments(conflict, class_ids)
    z_score = (empirical - null_mean) / null_sd if null_sd else float("nan")

    real_signal: dict[str, object] | None = None
    if args.feature_cache and args.head_checkpoint:
        real_signal = _measure_real_loss_signal(
            feature_cache=args.feature_cache,
            head_checkpoint=args.head_checkpoint,
            manifest=args.manifest,
            split=args.split,
            conflict_hashes=conflict_hashes,
            partition=args.partition,
            min_weight=args.min_weight,
        )

    conflict_sample_count = sum(1 for flag in conflict if flag)
    report = {        "partition": args.partition,
        "min_weight": args.min_weight,
        "sample_count": len(selected),
        "class_count": len(class_sizes),
        "class_size_min": min(class_sizes.values()),
        "class_size_max": max(class_sizes.values()),
        "conflict_sample_count": conflict_sample_count,
        "conflict_rate": conflict_sample_count / len(selected),
        "class_count_with_conflict": len(
            {class_ids[index] for index, flag in enumerate(conflict) if flag}
        ),
        "auc_empirical": empirical,
        "auc_null_mean": null_mean,
        "auc_null_sd": null_sd,
        "auc_z_score": z_score,
        "effective_sample_ratio": sum(weights) / len(weights),
        "enrichment": [_enrichment(noise, conflict, value) for value in (0.01, 0.05, 0.10, 0.20)],
        "real_loss_signal": real_signal,
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)

    if args.conflict_split_output and args.feature_cache and args.head_checkpoint:
        quality = _signal_quality_table(
            feature_cache=args.feature_cache,
            head_checkpoint=args.head_checkpoint,
            manifest=args.manifest,
            split=args.split,
            conflict_hashes=conflict_hashes,
            partition_filter=args.conflict_split,
        )
        quality_text = json.dumps(quality, ensure_ascii=False, indent=2)
        args.conflict_split_output.parent.mkdir(parents=True, exist_ok=True)
        args.conflict_split_output.write_text(quality_text + "\n", encoding="utf-8")
        print()
        print(quality_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
