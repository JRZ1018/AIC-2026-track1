"""Original single-model, training-only persistent supervision gate.

This module is NOT connected to a trainer by default. It does not delete images,
change labels, infer a true noise rate, or accept validation/test identities.
Inspired by small-loss curricula; no third-party implementation was copied.
"""
from __future__ import annotations

import math
from typing import Any

import torch


class PersistentTrainingGate:
    """Remove at most 10%/class from supervision after three persistent signals.

    All training examples still need observations every epoch so rejected examples
    can return. Call only after a complete AC epoch update; apply to the NEXT epoch.
    AC weights remain separate from this boolean mask and keep updating unchanged.
    """

    def __init__(self, sample_indices: torch.Tensor, labels: torch.Tensor):
        for x in (sample_indices, labels):
            if x.ndim != 1 or x.dtype != torch.long or x.device.type != 'cpu':
                raise ValueError('identities must be CPU long vectors')
        if len(labels) == 0 or sample_indices.shape != labels.shape:
            raise ValueError('nonempty matching training identities required')
        if bool((sample_indices < 0).any()) or bool((labels < 0).any()):
            raise ValueError('identities must be nonnegative')
        if torch.unique(sample_indices).numel() != len(labels):
            raise ValueError('duplicate training identity')
        self.sample_indices = sample_indices.clone()
        self.labels = labels.clone()
        self.streak = torch.zeros_like(labels)
        self.keep = torch.ones_like(labels, dtype=torch.bool)
        self.epochs_updated = 0
        self._lookup = torch.full((int(sample_indices.max()) + 1,), -1, dtype=torch.long)
        self._lookup[self.sample_indices] = torch.arange(len(labels))

    def keep_for(self, sample_indices):
        """Lookup only known train identities; validation/test IDs are rejected."""
        indices = sample_indices.detach().long().cpu()
        if indices.ndim != 1 or not len(indices):
            raise ValueError('nonempty training batch identities required')
        if bool((indices < 0).any()) or int(indices.max()) >= len(self._lookup):
            raise ValueError('unknown training identity')
        positions = self._lookup[indices]
        if bool((positions < 0).any()):
            raise ValueError('identity is outside training partition')
        return self.keep[positions]

    def update(self, sample_indices, weights, agreement, stability) -> dict[str, Any]:
        if not torch.equal(sample_indices, self.sample_indices):
            raise ValueError('training identity/order mismatch')
        signals = (weights, agreement, stability)
        if any(x.shape != self.labels.shape or x.device.type != 'cpu' for x in signals):
            raise ValueError('signals must match training identities on CPU')
        if any(not bool(torch.isfinite(x).all()) or bool(((x < 0) | (x > 1)).any()) for x in signals):
            raise ValueError('signals must be finite probabilities/weights')
        suspicious = (weights <= .10) & (agreement <= .20) & (stability >= .80)
        self.streak = torch.where(suspicious, self.streak + 1, 0)
        self.epochs_updated += 1
        self.keep.fill_(True)
        class_rows = []
        for label in torch.unique(self.labels, sorted=True):
            indices = torch.nonzero(self.labels == label, as_tuple=False).flatten()
            count = len(indices)
            # A floor is not a presumed clean count; tiny classes remain untouched.
            cap = min(math.floor(.10 * count), max(0, count - 16))
            eligible = indices[self.streak[indices] >= 3]
            # Ties resolve by explicit sample identity, not incoming batch order.
            order = sorted(eligible.tolist(), key=lambda i: (float(weights[i]), int(self.sample_indices[i])))
            original_mass = float(weights[indices].double().sum())
            removed_mass = 0.0
            removed = 0
            for i in order:
                if removed >= cap:
                    break
                mass = float(weights[i])
                if removed_mass + mass > .10 * original_mass:
                    continue
                self.keep[i] = False
                removed_mass += mass
                removed += 1
            class_rows.append(dict(label=int(label),count=count,excluded=removed,retained=count-removed,
                                   original_ac_mass=original_mass,retained_ac_mass=original_mass-removed_mass))
        return dict(epochs_updated=self.epochs_updated,excluded=int((~self.keep).sum()),
                    sample_count=len(self.keep),classes=class_rows,
                    note='suspected persistent disagreement, not measured true noise')

    def state_dict(self):
        return dict(protocol='persistent-supervision-gate-v1',sample_indices=self.sample_indices.clone(),
                    labels=self.labels.clone(),streak=self.streak.clone(),keep=self.keep.clone(),
                    epochs_updated=self.epochs_updated)

    def load_state_dict(self, state):
        if state.get('protocol') != 'persistent-supervision-gate-v1':
            raise ValueError('gate protocol mismatch')
        for name in ('sample_indices', 'labels'):
            if not torch.equal(state[name], getattr(self, name)):
                raise ValueError('gate training identities mismatch')
        streak, keep = state['streak'], state['keep']
        if streak.shape != self.labels.shape or streak.dtype != torch.long or bool((streak < 0).any()):
            raise ValueError('invalid gate streak')
        if keep.shape != self.labels.shape or keep.dtype != torch.bool:
            raise ValueError('invalid gate mask')
        epoch = state['epochs_updated']
        if not isinstance(epoch, int) or epoch < 0 or bool((streak > epoch).any()):
            raise ValueError('invalid gate epoch')
        if bool(((~keep) & (streak < 3)).any()):
            raise ValueError('restored exclusion lacks persistent observations')
        for label in torch.unique(self.labels):
            mask = self.labels == label
            count = int(mask.sum())
            if int((~keep[mask]).sum()) > min(math.floor(.10 * count), max(0, count - 16)):
                raise ValueError('restored mask violates class coverage')
        self.streak = streak.clone().cpu()
        self.keep = keep.clone().cpu()
        self.epochs_updated = epoch


def gated_weighted_loss(losses, weights, keep):
    """Return (loss, has_supervision); caller MUST skip optimizer if false.

    Every sample's detached raw prediction still enters the AC observation update.
    An all-excluded batch must not silently receive AdamW weight-decay updates.
    """
    if losses.ndim != 1 or losses.shape != weights.shape or keep.shape != losses.shape:
        raise ValueError('matching per-sample vectors required')
    if keep.dtype != torch.bool or not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
        raise ValueError('positive finite AC weights and boolean keep required')
    if not bool(torch.isfinite(losses).all()):
        raise ValueError('nonfinite losses')
    effective = weights.to(losses) * keep.to(device=losses.device)
    if not bool(keep.any()):
        return losses.sum() * 0, False
    return (losses * effective).sum() / effective.sum(), True
