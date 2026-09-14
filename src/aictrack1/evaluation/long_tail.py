"""Head/mid/tail, worst-class recall and per-class retention diagnostics.

The functions in this module only consume explicitly supplied labels, class
counts and class-to-stratum assignments. They never read competition test data
and never infer strata from file paths.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

HEAD = "head"
MID = "mid"
TAIL = "tail"
STRATA: tuple[str, str, str] = (HEAD, MID, TAIL)


@dataclass(frozen=True)
class Stratification:
    """Deterministic head/mid/tail assignment derived from per-class sample counts."""

    num_classes: int
    head_count: int
    mid_count: int
    class_counts: tuple[int, ...]
    assignments: tuple[str, ...]
    seed: int
    ordered_classes: tuple[int, ...]

    @property
    def counts(self) -> dict[str, int]:
        return {
            HEAD: self.head_count,
            MID: self.mid_count,
            TAIL: self.num_classes - self.head_count - self.mid_count,
        }


def stratify_classes(
    class_counts: Sequence[int] | torch.Tensor,
    *,
    head_fraction: float,
    mid_fraction: float,
    seed: int,
) -> Stratification:
    """Split classes into head/mid/tail strata by descending sample count.

    Ranking ties are broken by class index so the assignment never depends on the
    manifest row order or on file names. Head and mid keep at least one class, and
    the split stays contiguous in the sorted order.
    """
    if not 0 < head_fraction < 1 or not 0 < mid_fraction < 1:
        raise ValueError("head_fraction and mid_fraction must be strictly between 0 and 1")
    if head_fraction + mid_fraction >= 1:
        raise ValueError("head_fraction + mid_fraction must stay below 1")
    values = [int(value) for value in class_counts]
    if not values:
        raise ValueError("class counts cannot be empty")
    if any(value < 0 for value in values):
        raise ValueError("class counts cannot be negative")
    num_classes = len(values)
    if num_classes < 3:
        raise ValueError("head/mid/tail stratification requires at least three classes")
    ordered = tuple(
        sorted(range(num_classes), key=lambda index: (-values[index], index))
    )
    head_count = max(1, round(num_classes * head_fraction))
    mid_count = max(1, round(num_classes * mid_fraction))
    if head_count + mid_count >= num_classes:
        mid_count = num_classes - head_count - 1
    if mid_count < 1:
        raise ValueError("stratification leaves no mid class; check the fractions")
    assignments = [""] * num_classes
    for position, index in enumerate(ordered):
        if position < head_count:
            assignments[index] = HEAD
        elif position < head_count + mid_count:
            assignments[index] = MID
        else:
            assignments[index] = TAIL
    return Stratification(
        num_classes=num_classes,
        head_count=head_count,
        mid_count=mid_count,
        class_counts=tuple(values),
        assignments=tuple(assignments),
        seed=int(seed),
        ordered_classes=ordered,
    )


def write_stratification(
    stratification: Stratification,
    path: str | Path,
    *,
    retention_rate_by_class: Mapping[int, float] | None = None,
) -> None:
    """Persist the class-to-stratum assignment as an auditable CSV."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    retention = retention_rate_by_class or {}
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["internal_class_index", "reference_count", "stratum", "design_retention_rate"]
        )
        for index in range(stratification.num_classes):
            writer.writerow(
                [
                    index,
                    stratification.class_counts[index],
                    stratification.assignments[index],
                    f"{float(retention.get(index, 0.0)):.6f}",
                ]
            )


def read_stratification(path: str | Path) -> Stratification:
    """Reload a persisted stratification without recomputing it."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"internal_class_index", "reference_count", "stratum"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"stratification file requires columns {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("stratification file is empty")
    ordered = sorted(rows, key=lambda row: int(row["internal_class_index"]))
    indices = [int(row["internal_class_index"]) for row in ordered]
    if indices != list(range(len(indices))):
        raise ValueError("stratification class indices must be contiguous from zero")
    strata = [str(row["stratum"]) for row in ordered]
    unknown = sorted(set(strata) - set(STRATA))
    if unknown:
        raise ValueError(f"unknown strata {unknown}")
    counts = tuple(int(row["reference_count"]) for row in ordered)
    return Stratification(
        num_classes=len(ordered),
        head_count=strata.count(HEAD),
        mid_count=strata.count(MID),
        class_counts=counts,
        assignments=tuple(strata),
        seed=0,
        ordered_classes=tuple(
            sorted(range(len(ordered)), key=lambda index: (-counts[index], index))
        ),
    )


def per_class_recall(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int,
    present_classes: Sequence[int] | torch.Tensor | None = None,
) -> tuple[dict[int, float | None], dict[int, int]]:
    """Return recall (``None`` when a class has no evaluation support) and support.

    Every class in ``range(num_classes)`` is reported by default; pass
    ``present_classes`` to restrict the report to a subset of indices.
    """
    predictions = predictions.detach().cpu().long().flatten()
    targets = targets.detach().cpu().long().flatten()
    if predictions.shape != targets.shape or predictions.numel() == 0:
        raise ValueError("predictions and targets must be non-empty vectors of equal shape")
    if int(predictions.min()) < 0 or int(targets.min()) < 0:
        raise ValueError("class indices must be non-negative")
    if int(predictions.max()) >= num_classes or int(targets.max()) >= num_classes:
        raise ValueError("class index exceeds num_classes")
    if present_classes is None:
        candidates = list(range(num_classes))
    else:
        candidates = [int(value) for value in present_classes]
        if any(value < 0 or value >= num_classes for value in candidates):
            raise ValueError("present_classes entry exceeds num_classes")
    total = torch.bincount(targets, minlength=num_classes)
    correct = torch.bincount(targets[predictions.eq(targets)], minlength=num_classes)
    support = {index: int(total[index]) for index in candidates}
    recall = {
        index: (float(correct[index]) / float(total[index]) if int(total[index]) > 0 else None)
        for index in candidates
    }
    return recall, support


def long_tail_diagnostics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int,
    stratification: Stratification,
    reference_class_counts: Sequence[int] | torch.Tensor | None = None,
    evaluated_class_counts: Sequence[int] | torch.Tensor | None = None,
    min_stratum_samples: int = 50,
) -> dict[str, Any]:
    """Aggregate accuracy per stratum, worst-class recall and retention rates.

    A stratum whose evaluation support falls below ``min_stratum_samples`` is
    flagged as unstable and excluded from the head−tail gap, so a sparse stratum
    can never masquerade as a hard failure.
    """
    if stratification.num_classes != num_classes:
        raise ValueError("stratification and num_classes disagree")
    if min_stratum_samples < 1:
        raise ValueError("min_stratum_samples must be at least one")
    predictions = predictions.detach().cpu().long().flatten()
    targets = targets.detach().cpu().long().flatten()
    recall, support = per_class_recall(
        predictions, targets, num_classes=num_classes
    )
    correct = predictions.eq(targets)

    def stratum_mask(stratum: str) -> torch.Tensor:
        return torch.tensor(
            [value == stratum for value in stratification.assignments], dtype=torch.bool
        )

    strata_payload: dict[str, Any] = {}
    for stratum in STRATA:
        mask = stratum_mask(stratum)
        evaluated = [
            index
            for index in range(num_classes)
            if mask[index] and support[index] > 0
        ]
        sample_mask = mask[targets]
        recalls = torch.tensor(
            [float(recall[index]) for index in evaluated if recall[index] is not None]
        )
        sufficient = int(sample_mask.sum()) >= int(min_stratum_samples)
        payload: dict[str, Any] = {
            "class_count": int(mask.sum()),
            "evaluated_class_count": len(evaluated),
            "unevaluated_class_count": int(mask.sum()) - len(evaluated),
            "covered_class_count": sum(
                1 for index in evaluated if float(recall[index]) > 0.0
            ),
            "zero_recall_class_count": sum(
                1 for index in evaluated if float(recall[index]) == 0.0
            ),
            "sample_count": int(sample_mask.sum()),
            "accuracy": (
                float(correct[sample_mask].float().mean())
                if bool(sample_mask.any())
                else None
            ),
            "evaluated_class_mean_accuracy": (
                float(recalls.mean()) if recalls.numel() and sufficient else None
            ),
            "worst_class_recall": (
                float(recalls.min()) if recalls.numel() and sufficient else None
            ),
            "worst_class_index": (
                int(evaluated[int(recalls.argmin())])
                if recalls.numel() and sufficient
                else None
            ),
            "worst_class_support": (
                int(support[evaluated[int(recalls.argmin())]])
                if recalls.numel() and sufficient
                else None
            ),
            "sufficient_support": sufficient,
        }
        if recalls.numel():
            quantiles = torch.quantile(recalls, torch.tensor([0.1, 0.5, 0.9]))
            payload["recall_p10"] = float(quantiles[0])
            payload["recall_median"] = float(quantiles[1])
            payload["recall_p90"] = float(quantiles[2])
        strata_payload[stratum] = payload

    evaluated_overall = [index for index in range(num_classes) if support[index] > 0]
    if not evaluated_overall:
        raise ValueError("no class has evaluation support")
    result: dict[str, Any] = {
        "sample_count": predictions.numel(),
        "top1": float(correct.float().mean()),
        "macro_accuracy": sum(
            float(recall[index]) for index in evaluated_overall
        )
        / len(evaluated_overall),
        "evaluated_class_count": len(evaluated_overall),
        "zero_recall_class_count": sum(
            1 for index in evaluated_overall if float(recall[index]) == 0.0
        ),
        "strata": strata_payload,
        "min_stratum_samples": int(min_stratum_samples),
        "unstable_strata": [
            stratum
            for stratum in STRATA
            if not strata_payload[stratum]["sufficient_support"]
        ],
        "head_tail_accuracy_gap": _gap(strata_payload, HEAD, TAIL),
        "head_tail_macro_gap": _macro_gap(strata_payload, HEAD, TAIL),
    }
    if reference_class_counts is not None:
        reference = [int(value) for value in reference_class_counts]
        if len(reference) != num_classes:
            raise ValueError("reference_class_counts must cover every class")
        evaluated_counts = (
            [int(value) for value in evaluated_class_counts]
            if evaluated_class_counts is not None
            else [0] * num_classes
        )
        if len(evaluated_counts) != num_classes:
            raise ValueError("evaluated_class_counts must cover every class")
        result["retention"] = retention_diagnostics(
            reference,
            evaluated_counts,
            stratification=stratification,
        )
    return result


def _gap(strata_payload: Mapping[str, Any], left: str, right: str) -> float | None:
    if not strata_payload[left]["sufficient_support"] or not strata_payload[right]["sufficient_support"]:
        return None
    left_value = strata_payload[left]["accuracy"]
    right_value = strata_payload[right]["accuracy"]
    if left_value is None or right_value is None:
        return None
    return float(left_value) - float(right_value)


def _macro_gap(strata_payload: Mapping[str, Any], left: str, right: str) -> float | None:
    if not strata_payload[left]["sufficient_support"] or not strata_payload[right]["sufficient_support"]:
        return None
    left_value = strata_payload[left]["evaluated_class_mean_accuracy"]
    right_value = strata_payload[right]["evaluated_class_mean_accuracy"]
    if left_value is None or right_value is None:
        return None
    return float(left_value) - float(right_value)


def support_stratified_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int,
    train_class_counts: Sequence[int],
    max_bucket_support: int = 16,
) -> dict[str, Any]:
    """Bucket classes by how many training samples they received.

    This is the starvation probe: if recall collapses exactly for the classes
    that hold very few training samples, class-frequency skew — or an automatic
    sample filter — is starving them, and the buckets make that visible.
    """
    counts = [int(value) for value in train_class_counts]
    if len(counts) != num_classes:
        raise ValueError("train_class_counts must cover every class")
    recall, support = per_class_recall(predictions, targets, num_classes=num_classes)
    buckets: list[tuple[str, int, int | None]] = [
        ("train_support_1", 1, 1),
        ("train_support_2_4", 2, 4),
        ("train_support_5_8", 5, 8),
        ("train_support_9_16", 9, 16),
        (f"train_support_{max_bucket_support + 1}_plus", max_bucket_support + 1, None),
    ]
    payload: dict[str, Any] = {}
    for name, low, high in buckets:
        classes = [
            index
            for index in range(num_classes)
            if counts[index] >= low and (high is None or counts[index] <= high)
        ]
        evaluated = [index for index in classes if support[index] > 0]
        recalls = [float(recall[index]) for index in evaluated if recall[index] is not None]
        payload[name] = {
            "class_count": len(classes),
            "evaluated_class_count": len(evaluated),
            "evaluation_sample_count": sum(support[index] for index in classes),
            "zero_recall_class_count": sum(1 for value in recalls if value == 0.0),
            "mean_recall": (sum(recalls) / len(recalls)) if recalls else None,
            "worst_class_recall": min(recalls) if recalls else None,
            "evaluation_recall": _pooled_recall(predictions, targets, classes),
        }
    return {
        "buckets": payload,
        "total_train_samples": sum(counts),
    }


def _pooled_recall(
    predictions: torch.Tensor, targets: torch.Tensor, classes: Sequence[int]
) -> float | None:
    if not classes:
        return None
    mask = torch.zeros(int(targets.max()) + 1, dtype=torch.bool)
    for index in classes:
        mask[index] = True
    selected = mask[targets]
    if not bool(selected.any()):
        return None
    correct = predictions.eq(targets)
    return float(correct[selected].float().mean())


def retention_diagnostics(
    reference_class_counts: Sequence[int],
    retained_class_counts: Sequence[int],
    *,
    stratification: Stratification,
) -> dict[str, Any]:
    """Per-class retention rates plus starvation alarms.

    ``reference_class_counts`` is the current-stage training-partition count that
    the automatic pipeline starts from; ``retained_class_counts`` is what survives
    long-tail subsampling or any training-time sample selection. A class is
    reported as starved when nothing survives, and as minority when exactly one
    sample survives.
    """
    if len(reference_class_counts) != len(retained_class_counts):
        raise ValueError("reference and retained class counts must have equal length")
    if stratification.num_classes != len(reference_class_counts):
        raise ValueError("stratification and class counts disagree")
    rates: list[float] = []
    starved: list[int] = []
    minority: list[int] = []
    for index, (reference, retained) in enumerate(
        zip(reference_class_counts, retained_class_counts, strict=True)
    ):
        reference = int(reference)
        retained = int(retained)
        if reference < 0 or retained < 0:
            raise ValueError("class counts cannot be negative")
        rates.append(1.0 if reference == 0 else retained / reference)
        if retained == 0:
            starved.append(index)
        elif retained == 1:
            minority.append(index)
    per_stratum: dict[str, Any] = {}
    for stratum in STRATA:
        indices = [
            index
            for index in range(stratification.num_classes)
            if stratification.assignments[index] == stratum
        ]
        values = torch.tensor([rates[index] for index in indices])
        per_stratum[stratum] = {
            "class_count": len(indices),
            "mean_rate": float(values.mean()) if values.numel() else None,
            "min_rate": float(values.min()) if values.numel() else None,
            "max_rate": float(values.max()) if values.numel() else None,
            "starved_class_count": sum(
                1 for index in indices if int(retained_class_counts[index]) == 0
            ),
            "minority_class_count": sum(
                1 for index in indices if int(retained_class_counts[index]) == 1
            ),
        }
    return {
        "class_count": len(rates),
        "mean_rate": sum(rates) / len(rates),
        "min_rate": min(rates),
        "starved_class_count": len(starved),
        "starved_classes": starved,
        "minority_class_count": len(minority),
        "minority_classes": minority,
        "per_stratum": per_stratum,
    }


def retention_gate(
    retention: Mapping[str, Any],
    *,
    min_class_rate: float,
    min_tail_to_head_ratio: float,
) -> dict[str, Any]:
    """Apply the pre-registered starvation gate to a retention report."""
    if not 0 < min_class_rate <= 1:
        raise ValueError("min_class_rate must be in (0, 1]")
    if min_tail_to_head_ratio < 0:
        raise ValueError("min_tail_to_head_ratio cannot be negative")
    head = retention["per_stratum"][HEAD]["mean_rate"]
    tail = retention["per_stratum"][TAIL]["mean_rate"]
    checks: dict[str, Any] = {
        "no_starved_class": {
            "expected": 0,
            "actual": int(retention["starved_class_count"]),
        },
        "min_per_class_rate": {
            "expected_min": float(min_class_rate),
            "actual": float(retention["min_rate"]),
        },
        "tail_to_head_rate_ratio": {
            "expected_min": float(min_tail_to_head_ratio),
            "actual": (
                None if not head or tail is None else float(tail) / float(head)
            ),
        },
    }
    passed = (
        checks["no_starved_class"]["actual"] == 0
        and checks["min_per_class_rate"]["actual"] >= float(min_class_rate)
        and checks["tail_to_head_rate_ratio"]["actual"] is not None
        and checks["tail_to_head_rate_ratio"]["actual"] >= float(min_tail_to_head_ratio)
    )
    return {"passed": bool(passed), "checks": checks}
