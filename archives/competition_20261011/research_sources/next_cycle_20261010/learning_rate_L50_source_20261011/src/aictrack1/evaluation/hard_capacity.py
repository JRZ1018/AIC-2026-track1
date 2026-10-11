"""Deterministic, train-validated hard class-capacity decoding."""

from __future__ import annotations

import heapq

import numpy as np
import torch


def _move_to_bound(
    scores: np.ndarray,
    predictions: np.ndarray,
    *,
    threshold: int,
    mode: str,
) -> int:
    """Greedily move the lowest-regret rows needed by one capacity phase.

    Each row's destinations are sorted by regret. A heap performs a stable
    k-way merge of those per-row lists, equivalent to the registered global
    order (regret, row, class) without materializing a global edge sort.
    """
    if mode not in {"floor", "ceiling"}:
        raise ValueError("mode must be floor or ceiling")
    counts = np.bincount(predictions, minlength=scores.shape[1]).astype(np.int64)
    donor_classes = np.flatnonzero(counts > threshold)
    target_classes = np.flatnonzero(counts < threshold)
    source_remaining = np.maximum(counts - threshold, 0)
    target_remaining = np.maximum(threshold - counts, 0)

    if mode == "floor":
        goal = int(target_remaining[target_classes].sum())
    else:
        goal = int(source_remaining[donor_classes].sum())
    if goal == 0:
        return 0
    if donor_classes.size == 0 or target_classes.size == 0:
        raise RuntimeError(f"{mode} phase has a deficit but no donor or receiver")

    donor_rows = np.flatnonzero(np.isin(predictions, donor_classes))
    if donor_rows.size < goal:
        raise RuntimeError(f"{mode} phase has fewer donor rows than required moves")
    edge_count = int(donor_rows.size) * int(target_classes.size)
    if edge_count > 30_000_000:
        raise RuntimeError(f"{mode} phase edge budget exceeded: {edge_count}")

    row_sources = predictions[donor_rows]
    regrets = (
        scores[donor_rows, row_sources, None]
        - scores[donor_rows[:, None], target_classes[None, :]]
    )
    if not np.isfinite(regrets).all():
        raise ValueError("non-finite logit regret")
    # target_classes is ascending; stable sorting preserves class-id ties.
    order = np.argsort(regrets, axis=1, kind="stable")
    heap: list[tuple[float, int, int, int, int]] = []
    for position, row in enumerate(donor_rows):
        target_position = int(order[position, 0])
        target = int(target_classes[target_position])
        heapq.heappush(
            heap,
            (
                float(regrets[position, target_position]),
                int(row),
                target,
                position,
                0,
            ),
        )

    moved = 0
    while heap and moved < goal:
        _, row, target, position, rank = heapq.heappop(heap)
        source = int(predictions[row])
        if source_remaining[source] <= 0:
            continue
        if target_remaining[target] <= 0:
            next_rank = rank + 1
            if next_rank < target_classes.size:
                next_position = int(order[position, next_rank])
                next_target = int(target_classes[next_position])
                heapq.heappush(
                    heap,
                    (
                        float(regrets[position, next_position]),
                        row,
                        next_target,
                        position,
                        next_rank,
                    ),
                )
            continue
        predictions[row] = target
        source_remaining[source] -= 1
        target_remaining[target] -= 1
        moved += 1

    if moved != goal:
        raise RuntimeError(f"{mode} phase stopped after {moved}/{goal} required moves")
    return moved


def decode_hard_capacity(
    adjusted_logits: torch.Tensor,
    *,
    lower: int,
    upper: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Decode one sample per row under per-class lower/upper capacities.

    The objective is the registered deterministic greedy regret rule, not a
    claim of globally optimal assignment. Calculations and tie-breaking use
    CPU float64 copies of the supplied logits.
    """
    if not isinstance(adjusted_logits, torch.Tensor):
        raise TypeError("adjusted_logits must be a torch.Tensor")
    if adjusted_logits.ndim != 2 or not adjusted_logits.is_floating_point():
        raise ValueError("adjusted_logits must be a floating [rows, classes] tensor")
    if not torch.isfinite(adjusted_logits).all():
        raise ValueError("adjusted_logits must be finite")
    rows, classes = adjusted_logits.shape
    if rows < 1 or classes < 2:
        raise ValueError("at least one row and two classes are required")
    if lower < 0 or upper < lower:
        raise ValueError("capacity bounds must satisfy 0 <= lower <= upper")
    if lower * classes > rows or upper * classes < rows:
        raise ValueError("capacity bounds cannot contain all rows")

    scores = (
        adjusted_logits.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .numpy()
    )
    initial = scores.argmax(axis=1).astype(np.int64, copy=False)
    predictions = initial.copy()
    initial_counts = np.bincount(initial, minlength=classes)
    floor_moves = _move_to_bound(scores, predictions, threshold=lower, mode="floor")
    ceiling_moves = _move_to_bound(scores, predictions, threshold=upper, mode="ceiling")
    final_counts = np.bincount(predictions, minlength=classes)
    if int(final_counts.sum()) != rows:
        raise RuntimeError("capacity decode changed the number of rows")
    if (final_counts < lower).any() or (final_counts > upper).any():
        raise RuntimeError("capacity decode did not satisfy all bounds")

    decoded = torch.from_numpy(predictions.copy()).to(adjusted_logits.device)
    report: dict[str, object] = {
        "rows": int(rows),
        "classes": int(classes),
        "lower": int(lower),
        "upper": int(upper),
        "floor_phase_moves": floor_moves,
        "ceiling_phase_moves": ceiling_moves,
        "total_moves": floor_moves + ceiling_moves,
        "initial_counts": [int(value) for value in initial_counts],
        "final_counts": [int(value) for value in final_counts],
        "quota_exact": bool(
            int(final_counts.sum()) == rows
            and (final_counts >= lower).all()
            and (final_counts <= upper).all()
        ),
    }
    return decoded, report
