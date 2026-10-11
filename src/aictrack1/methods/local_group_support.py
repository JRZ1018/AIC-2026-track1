"""Original audit-only train feature neighbourhood statistic.

No fitted detector, class prototype, weight policy, teacher or label repair.
Every query's complete duplicate component is excluded. Each distinct neighbour
component receives one vote; conflicting labels share that vote equally by row.
Cosine ties use lexicographic component order, never label-aware tie breaking.
"""
from collections import Counter
import hashlib
from typing import Callable, Sequence

import torch
import torch.nn.functional as F


def independent_queries(sample_ids: Sequence[str], targets: Sequence[int], *,
                        num_classes: int, seed: int = 20261003,
                        skip: int = 1, take: int = 4) -> list[int]:
    """Next fixed SHA-ranked rows per class; never repeat first-audit queries."""
    if len(sample_ids) != len(targets) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError('unique IDs bound to targets')
    if skip < 1 or take < 1 or num_classes < 1:
        raise ValueError('independent nonempty query selection')
    bins=[[] for _ in range(num_classes)]
    for i,(sid,label) in enumerate(zip(sample_ids,targets)):
        if not isinstance(sid,str) or not sid or type(label) is not int or not 0<=label<num_classes:
            raise ValueError('explicit ID/class mapping')
        bins[label].append(i)
    result=[]
    for rows in bins:
        ordered=sorted(rows,key=lambda i:(hashlib.sha256(f'{seed}:{sample_ids[i]}'.encode()).digest(),sample_ids[i]))
        result.extend(ordered[skip:skip+take])
    return result


def local_group_support(features: torch.Tensor, targets: torch.Tensor,
                        groups: Sequence[str], queries: Sequence[int], *,
                        num_classes: int, k: int = 16, query_batch: int = 8,
                        guard: Callable[[], None] | None = None) -> dict:
    if features.device.type != 'cpu' or targets.device.type != 'cpu':
        raise ValueError('bounded CPU audit only')
    if features.ndim != 2 or targets.shape != (features.shape[0],):
        raise ValueError('feature/target row shape')
    if targets.dtype != torch.long or len(groups) != len(targets):
        raise ValueError('integer target and component binding')
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError('finite floating features')
    if num_classes < 1 or ((targets < 0) | (targets >= num_classes)).any():
        raise ValueError('fixed class mapping')
    if not queries or len(set(queries)) != len(queries) or any(type(q) is not int or q < 0 or q >= len(targets) for q in queries):
        raise ValueError('unique valid query row indices')
    if any(not isinstance(g, str) or not g for g in groups):
        raise ValueError('explicit component IDs')
    names = sorted(set(groups))
    if type(k) is not int or not 1 <= k < len(names) or query_batch < 1:
        raise ValueError('enough distinct other components for fixed k')
    lookup = {name: i for i, name in enumerate(names)}
    codes = torch.tensor([lookup[g] for g in groups])
    labels = [Counter() for _ in names]
    for code, label in zip(codes.tolist(), targets.tolist()):
        labels[code][label] += 1
    totals = [sum(c.values()) for c in labels]
    x = features.detach().float()
    if (x.norm(dim=1) <= 0).any():
        raise ValueError('zero feature norm')
    x = F.normalize(x,dim=1)
    selected, support, similarities = [], [], []
    with torch.no_grad():
        for start in range(0,len(queries),query_batch):
            if guard is not None: guard()
            q = torch.tensor(queries[start:start+query_batch])
            cos = x[q] @ x.T
            component_cos = torch.full((len(q),len(names)),-torch.inf)
            component_cos.scatter_reduce_(1,codes.expand(len(q),-1),cos,reduce='amax',include_self=True)
            component_cos[torch.arange(len(q)),codes[q]] = -torch.inf
            top = component_cos.argsort(dim=1,descending=True,stable=True)[:,:k]
            for row, neighbours in zip(q.tolist(),top.tolist()):
                assert codes[row].item() not in neighbours
                selected.append(neighbours)
                support.append(sum(labels[g][int(targets[row])]/totals[g] for g in neighbours)/k)
            similarities.extend(component_cos.gather(1,top).tolist())
    return dict(component_names=names,neighbour_components=selected,
                target_support=torch.tensor(support),cosines=torch.tensor(similarities),
                queries=list(queries),k=k,optimizer_steps=0)
