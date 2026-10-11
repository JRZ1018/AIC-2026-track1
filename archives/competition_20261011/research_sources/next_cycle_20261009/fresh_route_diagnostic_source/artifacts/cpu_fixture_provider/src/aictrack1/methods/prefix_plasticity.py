"""Original train-only block grouping: preserve late-block LR, adapt prefix slowly.

Not N4 layer-wise decay and not full-visual fine-tuning. No upstream code copied.
"""
from __future__ import annotations

def prefix_parameter_groups(visual, learning_rate: float, *, prefix_scale=None):
    if learning_rate <= 0:
        raise ValueError('positive visual learning rate required')
    names = [(n, p) for n, p in visual.named_parameters() if p.requires_grad]
    if prefix_scale is None:
        if not names:
            raise ValueError('no trainable visual parameters')
        return [{'params': [p for _, p in names], 'lr': learning_rate}]
    if float(prefix_scale) != 0.1 or len(visual.transformer.resblocks) != 12:
        raise ValueError('preregistered 12-block/0.1 prefix contract required')
    prefix, late = [], []
    blocks = set()
    for name, parameter in names:
        pieces = name.split('.')
        if len(pieces) < 4 or pieces[:2] != ['transformer', 'resblocks']:
            raise ValueError('stem/projection/norm must remain frozen')
        index = int(pieces[2])
        if index not in range(12):
            raise ValueError('invalid visual block')
        blocks.add(index)
        (prefix if index < 4 else late).append(parameter)
    if blocks != set(range(12)) or not prefix or not late:
        raise ValueError('all twelve blocks must be enabled')
    if len({id(p) for p in prefix + late}) != len(names):
        raise ValueError('duplicate optimizer parameters')
    return [{'params': prefix, 'lr': learning_rate * .1},
            {'params': late, 'lr': learning_rate}]
