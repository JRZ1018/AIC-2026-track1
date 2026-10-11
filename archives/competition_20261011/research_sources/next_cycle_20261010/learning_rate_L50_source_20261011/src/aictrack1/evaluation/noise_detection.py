"""Known-ground-truth metrics for synthetic noisy-label detection."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence


def _binary_counts(truth: Sequence[bool], predicted: Sequence[bool]) -> tuple[int, int, int, int]:
    tp = sum(t and p for t, p in zip(truth, predicted, strict=True))
    fp = sum((not t) and p for t, p in zip(truth, predicted, strict=True))
    fn = sum(t and (not p) for t, p in zip(truth, predicted, strict=True))
    tn = len(truth) - tp - fp - fn
    return tp, fp, fn, tn


def _metrics(truth: Sequence[bool], predicted: Sequence[bool]) -> dict[str, float | int]:
    tp, fp, fn, tn = _binary_counts(truth, predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    clean_retention = tn / (tn + fp) if tn + fp else 0.0
    noisy_leakage = fn / (tp + fn) if tp + fn else 0.0
    keep_rate = sum(not value for value in predicted) / len(predicted) if predicted else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "clean_retention": clean_retention,
        "noisy_leakage": noisy_leakage,
        "keep_rate": keep_rate,
    }


def binary_auroc(truth: Sequence[bool], scores: Sequence[float]) -> float | None:
    if len(truth) != len(scores):
        raise ValueError("AUROC inputs must have equal length")
    positives = sum(truth)
    negatives = len(truth) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(enumerate(scores), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position][0]] = average_rank
        start = end
    positive_rank_sum = sum(rank for rank, is_positive in zip(ranks, truth, strict=True) if is_positive)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def evaluate_noise_detection(
    is_noisy: Sequence[bool],
    predicted_noisy: Sequence[bool],
    class_ids: Sequence[str],
    noise_scores: Sequence[float] | None = None,
) -> dict[str, object]:
    if not (len(is_noisy) == len(predicted_noisy) == len(class_ids)):
        raise ValueError("metric inputs must have equal length")
    overall = _metrics(is_noisy, predicted_noisy)
    if noise_scores is not None:
        overall["auroc"] = binary_auroc(is_noisy, noise_scores)
    by_class_indices: dict[str, list[int]] = defaultdict(list)
    for index, class_id in enumerate(class_ids):
        by_class_indices[str(class_id)].append(index)
    by_class = {
        class_id: _metrics(
            [is_noisy[index] for index in indices],
            [predicted_noisy[index] for index in indices],
        )
        for class_id, indices in sorted(by_class_indices.items())
    }
    counts = Counter(str(value) for value in class_ids)
    ordered = sorted(counts, key=lambda key: (-counts[key], key))
    thirds = max(1, (len(ordered) + 2) // 3)
    buckets = {"head": ordered[:thirds], "mid": ordered[thirds : 2 * thirds], "tail": ordered[2 * thirds :]}
    bucket_metrics = {}
    for name, labels in buckets.items():
        indices = [index for index, value in enumerate(class_ids) if str(value) in set(labels)]
        bucket_metrics[name] = _metrics(
            [is_noisy[index] for index in indices],
            [predicted_noisy[index] for index in indices],
        ) if indices else {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "clean_retention": 0.0, "noisy_leakage": 0.0, "keep_rate": 0.0}
    return {"overall": overall, "by_class": by_class, "frequency_buckets": bucket_metrics}
