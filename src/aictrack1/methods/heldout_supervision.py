"""Original train-only exclusion contracts for a supervision signal audit.

This is not a noise detector or training curriculum. It fits no models and edits
no labels. Exact and near-duplicate components are unioned before label-blind
fold assignment, so a held-out sample's connected neighbours cannot enter fit.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Mapping, Sequence


def grouped_folds(rows: Sequence[Mapping[str, object]], *, folds: int = 3,
                  seed: int = 20261007) -> tuple[list[int], list[str]]:
    if folds < 2 or not rows:
        raise ValueError('nonempty train rows and at least two folds required')
    ids = [str(r['sample_id']) for r in rows]
    if len(set(ids)) != len(ids) or any(not sid for sid in ids):
        raise ValueError('unique nonempty sample IDs required')
    if any(r.get('partition') != 'train' or r.get('source_split') != 'train' for r in rows):
        raise ValueError('only explicit fold-train rows permitted')
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, row in enumerate(rows):
        # SHA binds exact duplicates even if a published group ID is absent.
        for field in ('sha256', 'duplicate_group', 'near_duplicate_group'):
            value = str(row.get(field) or '')
            if not value:
                continue
            key = (field, value)
            if key in owners:
                a, b = find(i), find(owners[key])
                parent[a] = b
            else:
                owners[key] = i
    names = {}
    for i, sid in enumerate(ids):
        representative = find(i)
        names[representative] = min(sid, names.get(representative, sid))
    groups = [names[find(i)] for i in range(len(rows))]
    assignment = {
        name: int.from_bytes(hashlib.sha256(f'{seed}\0{name}'.encode()).digest(), 'big') % folds
        for name in set(groups)
    }
    return [assignment[name] for name in groups], groups


def exclusion_contract(rows: Sequence[Mapping[str, object]], assignments: Sequence[int],
                       groups: Sequence[str], *, heldout_fold: int,
                       num_classes: int) -> tuple[list[int], list[int], dict]:
    if not (len(rows) == len(assignments) == len(groups)) or num_classes < 1:
        raise ValueError('row/assignment/group lengths or class count invalid')
    if any(r.get('partition') != 'train' or r.get('source_split') != 'train' for r in rows):
        raise ValueError('fit and held-out rows must both be explicit train')
    if len({str(r['sample_id']) for r in rows}) != len(rows):
        raise ValueError('duplicate sample IDs')
    if any(type(value) is not int or value < 0 for value in assignments):
        raise ValueError('folds require nonnegative integers')
    fit = [i for i, value in enumerate(assignments) if value != heldout_fold]
    held = [i for i, value in enumerate(assignments) if value == heldout_fold]
    if not fit or not held:
        raise ValueError('empty fit or held-out fold; no automatic reassignment')
    if set(groups[i] for i in fit) & set(groups[i] for i in held):
        raise ValueError('held-out duplicate component leaked into fitting')
    for field in ('sha256', 'duplicate_group', 'near_duplicate_group'):
        keys = lambda positions: {str(rows[i].get(field) or '') for i in positions} - {''}
        if keys(fit) & keys(held):
            raise ValueError('held-out duplicate edge leaked despite supplied component IDs')
    labels = [int(r['internal_class_index']) for r in rows]
    if any(c < 0 or c >= num_classes for c in labels):
        raise ValueError('class label outside fixed mapping')
    counts = Counter(labels[i] for i in fit)
    missing = [c for c in range(num_classes) if not counts[c]]
    if missing:
        raise ValueError(f'classes absent from fit {missing}; abstain, never include held-out labels')
    held_counts = Counter(labels[i] for i in held)
    digest = lambda positions: hashlib.sha256('\n'.join(sorted(str(rows[i]['sample_id']) for i in positions)).encode()).hexdigest()
    return fit, held, dict(fit_count=len(fit), heldout_count=len(held),
                           fit_ids_sha256=digest(fit), heldout_ids_sha256=digest(held),
                           group_overlap=0, fit_class_counts=[counts[c] for c in range(num_classes)],
                           heldout_class_counts=[held_counts[c] for c in range(num_classes)],
                           labels_used_to_assign_folds=False, small_class_policy='abstain if no independent fit support')
