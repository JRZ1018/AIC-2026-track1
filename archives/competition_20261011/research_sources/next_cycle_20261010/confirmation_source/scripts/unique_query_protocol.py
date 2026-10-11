"""Verbatim reviewed support-only sampling; frozen seed20261010, no predictions."""
import numpy as np

def unique_balanced_indices(targets, groups, per_class=20, seed=20261009):
    """Closed-set stress subtask, no row replacement, at most one per group.

    Inclusion uses observed-label/group support BEFORE candidate predictions.
    Shared conflicting groups may make a class ineligible. Ascending support
    priority is fixed; omitted classes are always disclosed.
    """
    targets = np.asarray(targets, dtype=np.int64)
    if len(groups) != len(targets) or per_class < 1:
        raise ValueError('invalid sampling input')
    rng = np.random.default_rng(seed)
    candidates = []
    for label in np.unique(targets):
        members = np.flatnonzero(targets == label)
        support = len({groups[i] for i in members})
        if support >= per_class:
            candidates.append((support, int(label), members))
    used = set()
    selected, retained = [], []
    for _, label, members in sorted(candidates, key=lambda item: item[:2]):
        proposed, current_groups = [], set()
        for i in rng.permutation(members):
            group = groups[int(i)]
            if group not in used and group not in current_groups:
                proposed.append(int(i))
                current_groups.add(group)
                if len(proposed) == per_class:
                    break
        if len(proposed) == per_class:
            selected.extend(proposed)
            retained.append(label)
            used.update(current_groups)
    if len(retained) < 2:
        raise ValueError('fewer than two classes have sufficient disjoint unique support')
    return np.asarray(selected, dtype=np.int64), np.asarray(sorted(retained), dtype=np.int64)
