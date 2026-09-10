"""Training-only, class-aware confidence utilities for noisy-label experiments."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F


def classwise_rank_noise_scores(losses: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
    """Map losses to tie-aware within-class percentile scores; higher means noisier."""
    if losses.ndim != 1 or class_ids.ndim != 1 or losses.shape != class_ids.shape:
        raise ValueError("losses and class_ids must be matching 1D tensors")
    if not torch.isfinite(losses).all():
        raise ValueError("losses must be finite")
    scores = torch.zeros_like(losses, dtype=torch.float32)
    for class_id in torch.unique(class_ids, sorted=True):
        indices = torch.nonzero(class_ids == class_id, as_tuple=False).flatten()
        values = losses.index_select(0, indices).float()
        if indices.numel() == 1:
            scores[indices] = 0.0
            continue
        order = torch.argsort(values, stable=True)
        sorted_values = values.index_select(0, order)
        ranks = torch.empty_like(sorted_values)
        start = 0
        while start < sorted_values.numel():
            end = start + 1
            while end < sorted_values.numel() and sorted_values[end] == sorted_values[start]:
                end += 1
            ranks[start:end] = (start + end - 1) / 2.0
            start = end
        percentiles = ranks / (indices.numel() - 1)
        scores[indices.index_select(0, order)] = percentiles
    return scores


def select_suspected_noisy(
    noise_scores: torch.Tensor,
    class_ids: torch.Tensor,
    *,
    fraction: float,
    min_keep_per_class: int = 1,
) -> torch.Tensor:
    """Select the highest scores within each observed class while protecting tails."""
    if noise_scores.ndim != 1 or class_ids.ndim != 1 or noise_scores.shape != class_ids.shape:
        raise ValueError("noise_scores and class_ids must be matching 1D tensors")
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0, 1]")
    if min_keep_per_class < 1:
        raise ValueError("min_keep_per_class must be at least 1")
    selected = torch.zeros_like(class_ids, dtype=torch.bool)
    for class_id in torch.unique(class_ids, sorted=True):
        indices = torch.nonzero(class_ids == class_id, as_tuple=False).flatten()
        remove_count = min(round(indices.numel() * fraction), max(0, indices.numel() - min_keep_per_class))
        if remove_count:
            order = torch.argsort(noise_scores.index_select(0, indices), descending=True, stable=True)
            selected[indices.index_select(0, order[:remove_count])] = True
    return selected


def weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Return a positive-weight normalized CE without changing target labels."""
    if logits.ndim != 2 or targets.ndim != 1 or weights.ndim != 1:
        raise ValueError("logits must be 2D and targets/weights must be 1D")
    if logits.shape[0] != targets.numel() or targets.shape != weights.shape:
        raise ValueError("logits, targets, and weights must share the batch dimension")
    if not torch.isfinite(weights).all() or bool((weights <= 0).any()):
        raise ValueError("weights must be finite and strictly positive")
    per_sample = F.cross_entropy(logits, targets, reduction="none")
    normalized = weights.to(device=per_sample.device, dtype=per_sample.dtype)
    return (per_sample * normalized).sum() / normalized.sum()


class ClasswiseHistoryWeighting:
    """Build next-epoch soft weights from classwise loss and prediction history.

    State is defined only for explicitly supplied training sample indices. Validation
    and test samples therefore cannot silently enter the weighting statistics.
    """

    def __init__(
        self,
        sample_indices: torch.Tensor,
        class_ids: torch.Tensor,
        *,
        momentum: float = 0.8,
        history_mix: float = 0.5,
        min_weight: float = 0.2,
    ) -> None:
        if sample_indices.ndim != 1 or class_ids.ndim != 1 or sample_indices.shape != class_ids.shape:
            raise ValueError("sample_indices and class_ids must be matching 1D tensors")
        if sample_indices.numel() == 0:
            raise ValueError("confidence state requires at least one training sample")
        if bool((sample_indices < 0).any()) or bool((class_ids < 0).any()):
            raise ValueError("sample and class indices must be non-negative")
        if torch.unique(sample_indices).numel() != sample_indices.numel():
            raise ValueError("training sample indices must be unique")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if not 0 <= history_mix <= 1:
            raise ValueError("history_mix must be in [0, 1]")
        if not 0 < min_weight <= 1:
            raise ValueError("min_weight must be in (0, 1]")

        self.sample_indices = sample_indices.detach().long().cpu().clone()
        self.class_ids = class_ids.detach().long().cpu().clone()
        self.momentum = float(momentum)
        self.history_mix = float(history_mix)
        self.min_weight = float(min_weight)
        self.epochs_updated = 0
        count = self.sample_indices.numel()
        self.weights = torch.ones(count, dtype=torch.float32)
        self.ema_observed_probability = torch.zeros(count, dtype=torch.float32)
        self.ema_label_agreement = torch.zeros(count, dtype=torch.float32)
        self.ema_prediction_stability = torch.zeros(count, dtype=torch.float32)
        self.previous_prediction = torch.full((count,), -1, dtype=torch.long)
        self.seen = torch.zeros(count, dtype=torch.bool)
        self._lookup = torch.full((int(self.sample_indices.max()) + 1,), -1, dtype=torch.long)
        self._lookup[self.sample_indices] = torch.arange(count)

    def _positions(self, sample_indices: torch.Tensor) -> torch.Tensor:
        indices = sample_indices.detach().long().cpu()
        if indices.ndim != 1:
            raise ValueError("sample_indices must be 1D")
        if indices.numel() and (bool((indices < 0).any()) or int(indices.max()) >= self._lookup.numel()):
            raise ValueError("unknown training sample index")
        positions = self._lookup.index_select(0, indices)
        if bool((positions < 0).any()):
            raise ValueError("unknown training sample index")
        return positions

    def weights_for(self, sample_indices: torch.Tensor) -> torch.Tensor:
        """Return the latest CPU float32 weights in the requested sample order."""
        return self.weights.index_select(0, self._positions(sample_indices))

    def update_epoch(
        self,
        sample_indices: torch.Tensor,
        losses: torch.Tensor,
        observed_probabilities: torch.Tensor,
        predictions: torch.Tensor,
    ) -> dict[str, float | int]:
        """Consume exactly one full training epoch and prepare next-epoch weights."""
        tensors = [sample_indices, losses, observed_probabilities, predictions]
        if any(tensor.ndim != 1 for tensor in tensors):
            raise ValueError("epoch confidence inputs must be 1D")
        if any(tensor.numel() != self.sample_indices.numel() for tensor in tensors):
            raise ValueError("epoch confidence update must cover every training sample exactly once")
        positions = self._positions(sample_indices)
        if torch.unique(positions).numel() != positions.numel():
            raise ValueError("epoch confidence update contains duplicate samples")
        if not torch.isfinite(losses).all() or not torch.isfinite(observed_probabilities).all():
            raise ValueError("losses and observed probabilities must be finite")
        if bool((observed_probabilities < 0).any()) or bool((observed_probabilities > 1).any()):
            raise ValueError("observed probabilities must be in [0, 1]")

        ordered_losses = torch.empty_like(self.weights)
        ordered_probabilities = torch.empty_like(self.weights)
        ordered_predictions = torch.empty_like(self.previous_prediction)
        ordered_losses[positions] = losses.detach().float().cpu()
        ordered_probabilities[positions] = observed_probabilities.detach().float().cpu()
        ordered_predictions[positions] = predictions.detach().long().cpu()

        agreement = (ordered_predictions == self.class_ids).float()
        stability = torch.where(
            self.seen,
            (ordered_predictions == self.previous_prediction).float(),
            torch.ones_like(agreement),
        )
        fresh = ~self.seen
        retained = self.momentum
        incoming = 1.0 - retained
        self.ema_observed_probability = torch.where(
            fresh,
            ordered_probabilities,
            retained * self.ema_observed_probability + incoming * ordered_probabilities,
        )
        self.ema_label_agreement = torch.where(
            fresh,
            agreement,
            retained * self.ema_label_agreement + incoming * agreement,
        )
        self.ema_prediction_stability = torch.where(
            fresh,
            stability,
            retained * self.ema_prediction_stability + incoming * stability,
        )
        self.previous_prediction.copy_(ordered_predictions)
        self.seen.fill_(True)

        history_trust = (
            self.ema_observed_probability
            * (0.5 + 0.5 * self.ema_label_agreement)
            * (0.75 + 0.25 * self.ema_prediction_stability)
        )
        current_noise = classwise_rank_noise_scores(ordered_losses, self.class_ids)
        history_noise = classwise_rank_noise_scores(-history_trust, self.class_ids)
        noise = (1.0 - self.history_mix) * current_noise + self.history_mix * history_noise
        self.weights = self.min_weight + (1.0 - self.min_weight) * (1.0 - noise)
        self.epochs_updated += 1
        return self.diagnostics()

    def diagnostics(self) -> dict[str, float | int]:
        quantiles = torch.quantile(self.weights, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0]))
        class_means = torch.stack(
            [self.weights[self.class_ids == class_id].mean() for class_id in torch.unique(self.class_ids)]
        )
        return {
            "epochs_updated": self.epochs_updated,
            "weight_min": float(quantiles[0]),
            "weight_p10": float(quantiles[1]),
            "weight_median": float(quantiles[2]),
            "weight_p90": float(quantiles[3]),
            "weight_max": float(quantiles[4]),
            "weight_mean": float(self.weights.mean()),
            "effective_sample_ratio": float(self.weights.sum() / self.weights.numel()),
            "class_mean_min": float(class_means.min()),
            "class_mean_max": float(class_means.max()),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "sample_indices": self.sample_indices.clone(),
            "class_ids": self.class_ids.clone(),
            "momentum": self.momentum,
            "history_mix": self.history_mix,
            "min_weight": self.min_weight,
            "epochs_updated": self.epochs_updated,
            "weights": self.weights.clone(),
            "ema_observed_probability": self.ema_observed_probability.clone(),
            "ema_label_agreement": self.ema_label_agreement.clone(),
            "ema_prediction_stability": self.ema_prediction_stability.clone(),
            "previous_prediction": self.previous_prediction.clone(),
            "seen": self.seen.clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for name, expected in (
            ("sample_indices", self.sample_indices),
            ("class_ids", self.class_ids),
        ):
            value = torch.as_tensor(state[name]).long().cpu()
            if not torch.equal(value, expected):
                raise ValueError(f"confidence checkpoint {name} mismatch")
        for name, expected in (
            ("momentum", self.momentum),
            ("history_mix", self.history_mix),
            ("min_weight", self.min_weight),
        ):
            if float(state[name]) != expected:
                raise ValueError(f"confidence checkpoint {name} mismatch")
        for name in (
            "weights",
            "ema_observed_probability",
            "ema_label_agreement",
            "ema_prediction_stability",
            "previous_prediction",
            "seen",
        ):
            target = getattr(self, name)
            value = torch.as_tensor(state[name], dtype=target.dtype).cpu()
            if value.shape != target.shape:
                raise ValueError(f"confidence checkpoint {name} shape mismatch")
            target.copy_(value)
        self.epochs_updated = int(state["epochs_updated"])
