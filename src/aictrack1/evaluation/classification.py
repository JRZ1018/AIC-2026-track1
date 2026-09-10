"""Classification diagnostics that operate only on explicitly supplied labels."""

from __future__ import annotations

from typing import Any

import torch


def classification_diagnostics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int,
    top_confusions: int = 20,
) -> dict[str, Any]:
    predictions = predictions.detach().cpu().long().flatten()
    targets = targets.detach().cpu().long().flatten()
    if predictions.shape != targets.shape or predictions.numel() == 0:
        raise ValueError("predictions and targets must be non-empty vectors of equal shape")
    if predictions.min() < 0 or targets.min() < 0:
        raise ValueError("class indices must be non-negative")
    if predictions.max() >= num_classes or targets.max() >= num_classes:
        raise ValueError("class index exceeds num_classes")

    correct = predictions.eq(targets)
    class_total = torch.bincount(targets, minlength=num_classes)
    class_correct = torch.bincount(targets[correct], minlength=num_classes)
    predicted_count = torch.bincount(predictions, minlength=num_classes)
    present = class_total > 0
    recall = class_correct[present].float() / class_total[present]
    errors = ~correct
    pairs = targets[errors] * num_classes + predictions[errors]
    pair_counts = torch.bincount(pairs, minlength=num_classes * num_classes)
    values, indices = torch.topk(pair_counts, k=min(top_confusions, int(errors.sum())))
    confusions = [
        {
            "target": int(index // num_classes),
            "prediction": int(index % num_classes),
            "count": int(value),
        }
        for value, index in zip(values, indices, strict=True)
        if int(value) > 0
    ]
    quantiles = torch.quantile(recall, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0]))
    return {
        "sample_count": predictions.numel(),
        "top1": float(correct.float().mean()),
        "macro_accuracy": float(recall.mean()),
        "present_target_classes": int(present.sum()),
        "predicted_classes": int((predicted_count > 0).sum()),
        "zero_recall_classes": int((recall == 0).sum()),
        "recall_quantiles": {
            "min": float(quantiles[0]),
            "p10": float(quantiles[1]),
            "median": float(quantiles[2]),
            "p90": float(quantiles[3]),
            "max": float(quantiles[4]),
        },
        "prediction_count_min": int(predicted_count.min()),
        "prediction_count_max": int(predicted_count.max()),
        "top_confusions": confusions,
    }
