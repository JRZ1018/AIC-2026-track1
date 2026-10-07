"""Original fixed grouped visual-supervision pool and matched exposure contracts.

No scoring, label repair, external data, or model fitting. SHA priorities never
use labels; class quotas use the existing noisy labels, not a cleanliness claim.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json

import torch
from torch.utils.data import Dataset

from .heldout_supervision import grouped_folds

VERSION = 'grouped-visual-pool-v1'


def bound_records(records, setting, *, root):
    """Load a source-frozen pool without consulting images/val/model scores."""
    from pathlib import Path
    if setting is None:
        return records, None
    if set(setting) != {'enabled', 'path', 'sha256'} or setting['enabled'] is not True:
        raise ValueError('invalid frozen training pool setting')
    relative = Path(setting['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('pool path must stay inside frozen scope')
    path = Path(root) / relative
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('pool path escapes frozen scope')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != setting['sha256']:
        raise ValueError('pool payload SHA mismatch')
    data = json.loads(raw)
    p = data['protocol']; ids = data['selected_ids']
    digest = hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    if (p['version'] != VERSION or p['full_count'] != len(records)
            or p['selected_count'] != len(ids) or p['selected_ids_sha256'] != digest
            or p['test_access'] is not False or len(set(ids)) != len(ids)):
        raise ValueError('pool identity or source length drift')
    wanted = set(ids)
    selected = [r for r in records if r.sample_id in wanted]
    if [r.sample_id for r in selected] != ids or any(r.partition != 'train' for r in records):
        raise ValueError('selected ID order/partition/source mismatch')
    counts = Counter(int(r.internal_class_index) for r in selected)
    full = Counter(int(r.internal_class_index) for r in records)
    if (p['class_counts'] != [counts[c] for c in range(len(p['class_counts']))]
            or p['full_class_counts'] != [full[c] for c in range(len(p['full_class_counts']))]):
        raise ValueError('class support metadata drift')
    return selected, p


def select_pool(rows, *, numerator=7, denominator=10, seed=20261007,
                num_classes=750, minimum_class_support=3):
    if (numerator, denominator) not in {(7, 10), (10, 10)}:
        raise ValueError('only preregistered 70/100 percent pools')
    if minimum_class_support != 3:
        raise ValueError('minimum support is frozen at three or all available')
    _, groups = grouped_folds(rows, seed=seed)
    labels = [int(r['internal_class_index']) for r in rows]
    if any(c < 0 or c >= num_classes for c in labels):
        raise ValueError('class outside mapping')
    full = Counter(labels)
    if any(full[c] == 0 for c in range(num_classes)):
        raise ValueError('full pool must cover every class')
    target = [max(min(3, full[c]), (numerator * full[c] + denominator // 2) // denominator)
              for c in range(num_classes)]
    members = defaultdict(list)
    for i, group in enumerate(groups):
        members[group].append(i)
    priority = lambda group: (hashlib.sha256(f'{seed}\0{group}'.encode()).digest(), group)
    count = Counter()
    chosen = set()
    for group in sorted(members, key=priority):
        group_count = Counter(labels[i] for i in members[group])
        if any(count[c] < target[c] for c in group_count):
            chosen.add(group)
            count.update(group_count)
    positions = [i for i, group in enumerate(groups) if group in chosen]
    excluded = set(groups) - chosen
    assert not chosen & excluded
    assert all(count[c] >= target[c] for c in range(num_classes))
    ids = [str(rows[i]['sample_id']) for i in positions]
    binding = [[str(r['sample_id']), labels[i], groups[i]] for i, r in enumerate(rows)]
    digest = lambda value: hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    protocol = dict(version=VERSION, numerator=numerator, denominator=denominator,
                    selection_seed=seed, full_count=len(rows), selected_count=len(ids),
                    selected_ids_sha256=digest(ids), row_group_binding_sha256=digest(binding),
                    class_counts=[count[c] for c in range(num_classes)],
                    full_class_counts=[full[c] for c in range(num_classes)], targets=target,
                    minimum_class_support=3, selected_group_count=len(chosen),
                    total_group_count=len(members), group_overlap=0,
                    selection='global label-blind SHA order; whole group if any class quota deficient',
                    sampling='private seeded repeated random permutation cycles; fixed full-count exposures',
                    ac='mean CE and label probability; modal prediction with smallest-index ties; once per selected ID',
                    excluded_ac='not in AC state and never observed',
                    augmentation='original seed/epoch/sample-index; repeated ID has same crop within epoch',
                    h1_scope='shared full fold train labels', test_access=False)
    return positions, protocol


def exposure_order(size, exposures, *, seed, epoch):
    if size < 1 or exposures < size or epoch < 0:
        raise ValueError('must expose every pool member at least once')
    # No use of global Python/NumPy/Torch/CUDA generators.
    digest = hashlib.sha256(f'{VERSION}\0{seed}\0{epoch}'.encode()).digest()
    rng = torch.Generator().manual_seed(int.from_bytes(digest[:8], 'big') & ((1 << 63) - 1))
    cycles = []
    remaining = exposures
    while remaining:
        cycle = torch.randperm(size, generator=rng)[:remaining]
        cycles.append(cycle)
        remaining -= len(cycle)
    return torch.cat(cycles)


class MatchedExposureDataset(Dataset):
    def __init__(self, dataset, *, exposures, seed, epoch):
        self.dataset = dataset
        self.order = exposure_order(len(dataset), exposures, seed=seed, epoch=epoch)

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        return self.dataset[int(self.order[index])]


def aggregate_observations(expected_ids, ids, losses, probabilities, predictions, *, num_classes):
    expected_ids = expected_ids.detach().long().cpu()
    ids = ids.detach().long().cpu()
    losses = losses.detach().float().cpu()
    probabilities = probabilities.detach().float().cpu()
    predictions = predictions.detach().long().cpu()
    if any(t.ndim != 1 for t in (expected_ids, ids, losses, probabilities, predictions)):
        raise ValueError('observations must be vectors')
    if len(set(expected_ids.tolist())) != len(expected_ids) or len(expected_ids) == 0:
        raise ValueError('unique nonempty expected IDs required')
    if not (len(ids) == len(losses) == len(probabilities) == len(predictions)):
        raise ValueError('observation lengths differ')
    if (not torch.isfinite(losses).all() or not torch.isfinite(probabilities).all()
            or (losses < 0).any() or (probabilities < 0).any() or (probabilities > 1).any()
            or (predictions < 0).any() or (predictions >= num_classes).any()):
        raise ValueError('nonfinite or invalid observations')
    sorted_ids, order = expected_ids.sort()
    position = torch.searchsorted(sorted_ids, ids)
    if (position >= len(sorted_ids)).any() or not torch.equal(sorted_ids[position], ids):
        raise ValueError('excluded/unknown ID was observed')
    position = order[position]
    counts = torch.bincount(position, minlength=len(expected_ids))
    if (counts == 0).any():
        raise ValueError('AC missing a selected ID; never fill with zeros')
    if (counts == 1).all():
        # Exact original once-per-ID semantics, not a numerically different sum.
        return ids, losses, probabilities, predictions, counts
    loss_sum = torch.zeros(len(expected_ids), dtype=torch.float64)
    prob_sum = torch.zeros_like(loss_sum)
    loss_sum.index_add_(0, position, losses.double())
    prob_sum.index_add_(0, position, probabilities.double())
    keys, votes = torch.unique(position * num_classes + predictions, sorted=True, return_counts=True)
    vote_rows = keys // num_classes
    modes = torch.empty(len(expected_ids), dtype=torch.long)
    # Segment reduction avoids dense N x 750 storage and per-ID Python loops.
    max_votes = torch.zeros(len(expected_ids), dtype=votes.dtype)
    max_votes.scatter_reduce_(0, vote_rows, votes, reduce='amax', include_self=True)
    winners = torch.where(votes == max_votes[vote_rows], keys % num_classes, num_classes)
    modes.fill_(num_classes)
    modes.scatter_reduce_(0, vote_rows, winners, reduce='amin', include_self=True)
    return expected_ids, (loss_sum / counts).float(), (prob_sum / counts).float(), modes, counts


def verify_protocol(saved, expected):
    if saved != expected:
        raise ValueError('frozen training pool protocol or sample binding drift')
