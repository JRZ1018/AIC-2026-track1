"""Original first-epoch curriculum; no teacher, label changes or extra AC epoch.

Reuse the project's existing rank formula on the initial model's train-only
predictions in an isolated temporary state. Only the resulting weights survive.
This is not a noise-rate estimator or a reproduction of a two-network method.
"""
from __future__ import annotations

from typing import Any, Mapping
import torch
from .confidence import ClasswiseHistoryWeighting


class FirstEpochCurriculum:
    SCHEMA = "aictrack1.initial_train_curriculum.v1"

    def __init__(self, sample_indices: torch.Tensor, class_ids: torch.Tensor,
                 weights: torch.Tensor, *, min_weight: float = .02) -> None:
        if sample_indices.dtype != torch.long or class_ids.dtype != torch.long:
            raise ValueError("indices and classes must be int64")
        self.index = ClasswiseHistoryWeighting(sample_indices, class_ids,
                                              min_weight=min_weight)
        if weights.shape != sample_indices.shape or not torch.isfinite(weights).all():
            raise ValueError("weights must cover the exact finite training rows")
        if bool((weights < min_weight).any()) or bool((weights > 1).any()):
            raise ValueError("curriculum weights outside registered range")
        self.index.weights.copy_(weights.detach().float().cpu())

    @classmethod
    def from_initial_statistics(cls, formal: ClasswiseHistoryWeighting,
                                sample_indices: torch.Tensor, losses: torch.Tensor,
                                probabilities: torch.Tensor, predictions: torch.Tensor):
        if formal.epochs_updated != 0 or bool(formal.seen.any()):
            raise ValueError("bootstrap only before any formal AC observation")
        if bool((losses < 0).any()) or predictions.dtype != torch.long:
            raise ValueError("invalid initial losses/predictions")
        if bool((predictions < 0).any()) or bool((predictions > int(formal.class_ids.max())).any()):
            raise ValueError("prediction outside registered classes")
        isolated = ClasswiseHistoryWeighting(
            formal.sample_indices, formal.class_ids, momentum=formal.momentum,
            history_mix=formal.history_mix, min_weight=formal.min_weight,
            magnitude_mix=formal.magnitude_mix,
            tail_class_fraction=formal.tail_class_fraction,
            tail_min_mean_weight=formal.tail_min_mean_weight,
            min_effective_mass_per_class=formal.min_effective_mass_per_class)
        isolated.update_epoch(sample_indices, losses, probabilities, predictions)
        result = cls(formal.sample_indices, formal.class_ids, isolated.weights,
                     min_weight=formal.min_weight)
        diagnostics = isolated.diagnostics()
        diagnostics.pop("epochs_updated")
        return result, dict(diagnostics, bootstrap_observations=1,
                            formal_epochs_updated=0, applied_epochs=[1],
                            noise_rate_estimated=False)

    def weights_for(self, sample_indices: torch.Tensor, *, epoch_index: int):
        if epoch_index < 0:
            raise ValueError("negative epoch")
        # Always verify IDs, even in the inactive path.
        values = self.index.weights_for(sample_indices)
        return values if epoch_index == 0 else None

    def state_dict(self) -> dict[str, Any]:
        return dict(schema=self.SCHEMA, sample_indices=self.index.sample_indices.clone(),
                    class_ids=self.index.class_ids.clone(), weights=self.index.weights.clone(),
                    min_weight=self.index.min_weight, applied_epoch=1,
                    formal_history_seeded=False)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (state.get("schema") != self.SCHEMA or state.get("applied_epoch") != 1
                or state.get("formal_history_seeded") is not False
                or state.get("min_weight") != self.index.min_weight):
            raise ValueError("curriculum protocol mismatch")
        for name, expected in (("sample_indices", self.index.sample_indices),
                               ("class_ids", self.index.class_ids)):
            if not torch.equal(state[name], expected):
                raise ValueError("curriculum row binding mismatch")
        valid = type(self)(state["sample_indices"], state["class_ids"], state["weights"],
                           min_weight=state["min_weight"])
        self.index.weights.copy_(valid.index.weights)
