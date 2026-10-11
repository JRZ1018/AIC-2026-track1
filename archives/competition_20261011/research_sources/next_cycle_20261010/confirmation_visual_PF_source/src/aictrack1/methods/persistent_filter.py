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

    def __init__(self, sample_indices: torch.Tensor, labels: torch.Tensor, *, max_ac_weight: float):
        if isinstance(max_ac_weight,bool) or max_ac_weight not in (.10,.30,.50): raise ValueError("PF2 .30/.50 or .10 mathematical-reference threshold only")
        self.max_ac_weight=float(max_ac_weight)
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
        self.last_ac_weights=torch.ones(len(labels),dtype=torch.float32)
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
        suspicious = (weights <= self.max_ac_weight) & (agreement <= .20) & (stability >= .80)
        self.streak = torch.where(suspicious, self.streak + 1, 0)
        self.epochs_updated += 1
        self.last_ac_weights=weights.detach().clone()
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
        return dict(protocol='persistent-supervision-gate-v2',max_ac_weight=self.max_ac_weight,last_ac_weights=self.last_ac_weights.clone(),sample_indices=self.sample_indices.clone(),
                    labels=self.labels.clone(),streak=self.streak.clone(),keep=self.keep.clone(),
                    epochs_updated=self.epochs_updated)

    def load_state_dict(self, state, *, ac_weights=None):
        if state.get('protocol') != 'persistent-supervision-gate-v2' or state.get('max_ac_weight') != self.max_ac_weight:
            raise ValueError('gate protocol mismatch')
        for name in ('sample_indices', 'labels'):
            if not torch.equal(state[name], getattr(self, name)):
                raise ValueError('gate training identities mismatch')
        streak, keep = state['streak'], state['keep']
        weights=state['last_ac_weights']
        if not isinstance(weights,torch.Tensor) or weights.shape!=self.labels.shape or weights.device.type!='cpu' or not weights.is_floating_point() or not bool(torch.isfinite(weights).all()) or bool(((weights<=0)|(weights>1)).any()): raise ValueError('invalid restored gate AC weights')
        if ac_weights is not None and not torch.equal(weights,ac_weights):raise ValueError('gate/AC restored weights mismatch')
        if streak.shape != self.labels.shape or streak.dtype != torch.long or bool((streak < 0).any()):
            raise ValueError('invalid gate streak')
        if keep.shape != self.labels.shape or keep.dtype != torch.bool:
            raise ValueError('invalid gate mask')
        epoch = state['epochs_updated']
        if not isinstance(epoch, int) or epoch < 0 or bool((streak > epoch).any()):
            raise ValueError('invalid gate epoch')
        if bool(((~keep) & ((streak < 3) | (weights > self.max_ac_weight))).any()):
            raise ValueError('restored exclusion lacks persistent observations')
        for label in torch.unique(self.labels):
            mask = self.labels == label
            count = int(mask.sum())
            if int((~keep[mask]).sum()) > min(math.floor(.10 * count), max(0, count - 16)):
                raise ValueError('restored mask violates class coverage')
            original_mass=float(weights[mask].double().sum());removed_mass=float(weights[mask & ~keep].double().sum())
            if removed_mass>.10*original_mass+1e-12:raise ValueError('restored mask violates AC mass cap')
        self.streak = streak.clone().cpu()
        self.keep = keep.clone().cpu()
        self.epochs_updated = epoch
        self.last_ac_weights=weights.clone()


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


class ActualSupervisionLedger:
    """Original loader coverage plus CPU snapshot of actual epoch mask/AC weights.

    Snapshot precedes later AC/gate updates. No RNG or extra model forward.
    Mass describes supervised weights, not gradient influence or noise precision.
    """
    def __init__(self, gate, weights):
        import hashlib
        if weights.shape!=gate.labels.shape or weights.device.type!='cpu' or not bool(torch.isfinite(weights).all()) or bool((weights<=0).any()):raise ValueError('invalid actual AC weights')
        self.indices=gate.sample_indices.clone();self.labels=gate.labels.clone();self.keep=gate.keep.clone();self.weights=weights.detach().clone();self.lookup=gate._lookup.clone();self.seen=torch.zeros_like(self.keep)
        self.mask_source_completed_epoch=gate.epochs_updated
        self.mask_sha256=hashlib.sha256(self.keep.numpy().tobytes(order='C')).hexdigest()
    def observe(self, indices, labels):
        indices=indices.detach().long().cpu();labels=labels.detach().long().cpu()
        if indices.ndim!=1 or indices.shape!=labels.shape or not len(indices) or bool((indices<0).any()) or int(indices.max())>=len(self.lookup):raise ValueError('actual unknown batch identity')
        positions=self.lookup[indices]
        if bool((positions<0).any()) or torch.unique(positions).numel()!=len(positions) or bool(self.seen[positions].any()) or not torch.equal(labels,self.labels[positions]):raise ValueError('actual duplicate/label/partition mismatch')
        self.seen[positions]=True
    def finish(self):
        if not bool(self.seen.all()):raise ValueError('actual supervision ledger incomplete epoch')
        rows=[]
        for label in torch.unique(self.labels,sorted=True):
            mask=self.labels==label;original=float(self.weights[mask].double().sum());removed=float(self.weights[mask & ~self.keep].double().sum())
            rows.append(dict(label=int(label),observed=int(mask.sum()),supervised=int((mask & self.keep).sum()),excluded=int((mask & ~self.keep).sum()),original_AC_mass=original,removed_AC_mass=removed,retained_AC_mass=original-removed,removed_AC_mass_fraction=removed/original))
        original=float(self.weights.double().sum());removed=float(self.weights[~self.keep].double().sum())
        return dict(mask_role='actual_epoch_supervision_mask',mask_source_completed_epoch=self.mask_source_completed_epoch,mask_sha256=self.mask_sha256,observed=len(self.indices),supervised=int(self.keep.sum()),excluded=int((~self.keep).sum()),original_AC_mass=original,removed_AC_mass=removed,retained_AC_mass=original-removed,removed_AC_mass_fraction=removed/original,classes=rows,definition='weighted mass before retained-weight batch renormalization; AMP skips separate')
