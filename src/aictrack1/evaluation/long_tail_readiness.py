"""Pre-registered long-tail readiness metrics and L0/L1/L2 decisions."""

from __future__ import annotations

import math
from typing import Any, Sequence


LEVELS = ("L0", "L1", "L2")
LEVEL_ORDER = {level: index for index, level in enumerate(LEVELS)}
DECISION_SCHEMA = "aictrack1.long_tail_readiness.v1"
THRESHOLDS = {
    "support": {
        "l1_min_to_median_below": 0.50,
        "l1_p10_to_median_below": 0.70,
        "l1_max_to_min_above": 3.0,
        "l1_gini_above": 0.25,
        "l2_min_to_median_below": 0.10,
        "l2_p10_to_median_below": 0.25,
        "l2_max_to_min_above": 20.0,
        "l2_unsafe_split_fraction_at_least": 0.05,
    },
    "effective_quality": {
        "l1_min_ratio_below": 0.35,
        "l1_p10_ratio_below": 0.50,
        "l2_min_ratio_below": 0.10,
        "l2_p10_ratio_below": 0.25,
    },
}


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile input cannot be empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must be between zero and one")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def _gini(values: Sequence[int]) -> float:
    ordered = sorted(float(value) for value in values)
    total = sum(ordered)
    if total == 0.0:
        return 0.0
    count = len(ordered)
    weighted = sum((2 * index - count - 1) * value for index, value in enumerate(ordered, 1))
    return weighted / (count * total)


def _effective_number(values: Sequence[int]) -> float:
    total = float(sum(values))
    squared = sum(float(value) ** 2 for value in values)
    return total * total / squared if squared > 0.0 else 0.0


def _at_least(current: str, candidate: str) -> str:
    return candidate if LEVEL_ORDER[candidate] > LEVEL_ORDER[current] else current


def support_distribution_profile(
    class_counts: Sequence[int],
    *,
    validation_fraction: float = 0.2,
    min_train_per_class: int = 1,
) -> dict[str, Any]:
    """Summarize class support and apply frozen relative L0/L1/L2 thresholds."""

    counts = [int(value) for value in class_counts]
    if not counts:
        raise ValueError("class_counts cannot be empty")
    if any(value < 0 for value in counts):
        raise ValueError("class_counts cannot contain negative values")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between zero and one")
    if min_train_per_class < 1:
        raise ValueError("min_train_per_class must be positive")
    ordered = sorted(float(value) for value in counts)
    minimum = int(ordered[0])
    maximum = int(ordered[-1])
    p1 = _quantile(ordered, 0.01)
    p10 = _quantile(ordered, 0.10)
    median = _quantile(ordered, 0.50)
    p90 = _quantile(ordered, 0.90)
    zero_classes = [index for index, value in enumerate(counts) if value == 0]
    unsafe = [
        index
        for index, value in enumerate(counts)
        if value < min_train_per_class + 1
    ]
    max_to_min = None if minimum == 0 else maximum / minimum
    min_to_median = None if median == 0 else minimum / median
    p10_to_median = None if median == 0 else p10 / median
    unsafe_fraction = len(unsafe) / len(counts)
    triggers: list[dict[str, Any]] = []
    level = "L0"

    def trigger(candidate: str, metric: str, actual: Any, threshold: Any) -> None:
        nonlocal level
        level = _at_least(level, candidate)
        triggers.append(
            {"level": candidate, "metric": metric, "actual": actual, "threshold": threshold}
        )

    support_thresholds = THRESHOLDS["support"]
    if zero_classes:
        trigger("L2", "zero_class_count", len(zero_classes), 0)
    if min_to_median is not None:
        if min_to_median < support_thresholds["l2_min_to_median_below"]:
            trigger("L2", "min_to_median", min_to_median, support_thresholds["l2_min_to_median_below"])
        elif min_to_median < support_thresholds["l1_min_to_median_below"]:
            trigger("L1", "min_to_median", min_to_median, support_thresholds["l1_min_to_median_below"])
    if p10_to_median is not None:
        if p10_to_median < support_thresholds["l2_p10_to_median_below"]:
            trigger("L2", "p10_to_median", p10_to_median, support_thresholds["l2_p10_to_median_below"])
        elif p10_to_median < support_thresholds["l1_p10_to_median_below"]:
            trigger("L1", "p10_to_median", p10_to_median, support_thresholds["l1_p10_to_median_below"])
    if max_to_min is not None:
        if max_to_min > support_thresholds["l2_max_to_min_above"]:
            trigger("L2", "max_to_min", max_to_min, support_thresholds["l2_max_to_min_above"])
        elif max_to_min > support_thresholds["l1_max_to_min_above"]:
            trigger("L1", "max_to_min", max_to_min, support_thresholds["l1_max_to_min_above"])
    gini = _gini(counts)
    if gini > support_thresholds["l1_gini_above"]:
        trigger("L1", "gini", gini, support_thresholds["l1_gini_above"])
    if unsafe_fraction >= support_thresholds["l2_unsafe_split_fraction_at_least"] and unsafe:
        trigger(
            "L2",
            "unsafe_split_fraction",
            unsafe_fraction,
            support_thresholds["l2_unsafe_split_fraction_at_least"],
        )
    elif unsafe:
        trigger("L1", "unsafe_split_class_count", len(unsafe), 0)
    return {
        "schema": DECISION_SCHEMA,
        "decision": level,
        "class_count": len(counts),
        "sample_count": sum(counts),
        "support": {
            "min": minimum,
            "p1": p1,
            "p10": p10,
            "median": median,
            "p90": p90,
            "max": maximum,
            "max_to_min": max_to_min,
            "min_to_median": min_to_median,
            "p10_to_median": p10_to_median,
            "gini": gini,
            "effective_number": _effective_number(counts),
        },
        "zero_sample_classes": zero_classes,
        "unsafe_split_classes": unsafe,
        "unsafe_split_fraction": unsafe_fraction,
        "split_assumptions": {
            "validation_fraction": validation_fraction,
            "min_train_per_class": min_train_per_class,
            "requires_at_least_one_validation_sample": True,
        },
        "thresholds": support_thresholds,
        "triggers": triggers,
    }


def effective_training_quality_profile(
    labels: Sequence[int],
    losses: Sequence[float],
    weights: Sequence[float],
    *,
    num_classes: int,
    downweight_threshold: float = 0.5,
    sampling_counts: Sequence[int] | None = None,
    recalls: Sequence[float | None] | None = None,
) -> dict[str, Any]:
    """Separate raw support from quality lost through weighting or sampling."""

    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    if not (len(labels) == len(losses) == len(weights)) or not labels:
        raise ValueError("labels, losses and weights must be non-empty equal-length sequences")
    if not 0.0 <= downweight_threshold <= 1.0:
        raise ValueError("downweight_threshold must be between zero and one")
    if any(index < 0 or index >= num_classes for index in labels):
        raise ValueError("label exceeds num_classes")
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in losses):
        raise ValueError("losses must be finite and non-negative")
    if any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in weights):
        raise ValueError("weights must be finite values between zero and one")
    if sampling_counts is not None and len(sampling_counts) != num_classes:
        raise ValueError("sampling_counts length must equal num_classes")
    if recalls is not None and len(recalls) != num_classes:
        raise ValueError("recalls length must equal num_classes")
    rows: list[dict[str, Any]] = []
    for class_index in range(num_classes):
        positions = [position for position, label in enumerate(labels) if label == class_index]
        support = len(positions)
        total_weight = sum(float(weights[position]) for position in positions)
        mean_weight = total_weight / support if support else None
        mean_loss = (
            sum(float(losses[position]) for position in positions) / support if support else None
        )
        downweighted = sum(
            1 for position in positions if float(weights[position]) < downweight_threshold
        )
        recall = None if recalls is None else recalls[class_index]
        rows.append(
            {
                "internal_class_index": class_index,
                "raw_support": support,
                "mean_loss": mean_loss,
                "mean_weight": mean_weight,
                "effective_quality": total_weight,
                "effective_to_raw_ratio": total_weight / support if support else None,
                "downweighted_count": downweighted,
                "downweighted_fraction": downweighted / support if support else None,
                "sampling_count": support if sampling_counts is None else int(sampling_counts[class_index]),
                "recall": recall,
                "zero_recall": bool(recall == 0.0) if recall is not None else None,
            }
        )
    ratios = sorted(
        float(row["effective_to_raw_ratio"])
        for row in rows
        if row["effective_to_raw_ratio"] is not None
    )
    thresholds = THRESHOLDS["effective_quality"]
    minimum = min(ratios) if ratios else None
    p10 = _quantile(ratios, 0.10) if ratios else None
    level = "L0"
    triggers: list[dict[str, Any]] = []

    def trigger(candidate: str, metric: str, actual: float, threshold: float) -> None:
        nonlocal level
        level = _at_least(level, candidate)
        triggers.append(
            {"level": candidate, "metric": metric, "actual": actual, "threshold": threshold}
        )

    if minimum is not None:
        if minimum < thresholds["l2_min_ratio_below"]:
            trigger("L2", "min_effective_to_raw_ratio", minimum, thresholds["l2_min_ratio_below"])
        elif minimum < thresholds["l1_min_ratio_below"]:
            trigger("L1", "min_effective_to_raw_ratio", minimum, thresholds["l1_min_ratio_below"])
    if p10 is not None:
        if p10 < thresholds["l2_p10_ratio_below"]:
            trigger("L2", "p10_effective_to_raw_ratio", p10, thresholds["l2_p10_ratio_below"])
        elif p10 < thresholds["l1_p10_ratio_below"]:
            trigger("L1", "p10_effective_to_raw_ratio", p10, thresholds["l1_p10_ratio_below"])
    return {
        "schema": DECISION_SCHEMA,
        "decision": level,
        "class_count": num_classes,
        "observation_count": len(labels),
        "downweight_threshold": downweight_threshold,
        "summary": {
            "min_effective_to_raw_ratio": minimum,
            "p10_effective_to_raw_ratio": p10,
            "zero_recall_class_count": sum(1 for row in rows if row["zero_recall"] is True),
        },
        "thresholds": thresholds,
        "triggers": triggers,
        "classes": rows,
    }


def combined_long_tail_decision(
    support_profile: dict[str, Any],
    quality_profile: dict[str, Any] | None = None,
) -> str:
    level = str(support_profile["decision"])
    if level not in LEVEL_ORDER:
        raise ValueError(f"unknown support decision {level!r}")
    if quality_profile is not None:
        quality_level = str(quality_profile["decision"])
        if quality_level not in LEVEL_ORDER:
            raise ValueError(f"unknown quality decision {quality_level!r}")
        level = _at_least(level, quality_level)
    return level
