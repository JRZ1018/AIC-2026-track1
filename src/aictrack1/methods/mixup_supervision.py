"""Original train-only input mixing from arXiv:1710.09412v2 ideas.

No author source copied (CC-BY-NC-4.0); no pseudo-label refurbishment.
Weighted SCE is a convex sum of TWO hard-label losses, not soft-label RCE.
"""
from __future__ import annotations

import hashlib
import math
import random

import torch

from aictrack1.methods.sce import symmetric_cross_entropy

PROTOCOL = {
    'schema': 'aictrack1.supervised_input_mix.v1',
    'alpha': .2,
    'lambda_policy': 'max(beta,1-beta)',
    'pairing': 'batch_permutation_private_rng',
    'scope': 'stage2_fold_train_only',
    'ac_observation': 'original_strong_once_per_epoch',
    'teacher': False,
    'label_refurbishment': False,
}


def verify_protocol_state(state, setting=PROTOCOL):
    if state != setting or (setting is not None and setting != PROTOCOL):
        raise ValueError('input mixing protocol drift')


def protocol_state(setting):
    verify_protocol_state(setting, setting)
    return dict(setting) if setting is not None else None


def add_current_mix_loss(supervised, mixed_logits, targets, weights, order, coefficient, setting):
    verify_protocol_state(setting, setting)
    if setting is None:
        return supervised, None
    value = mixed_weighted_sce(mixed_logits, targets, weights, order, coefficient)
    return supervised + value, value


def batch_plan(count, seed, epoch, step):
    if any(type(v) is not int or v < 0 for v in (count, seed, epoch, step)) or count < 1:
        raise ValueError('positive batch and nonnegative integer seed/epoch/step required')
    digest = hashlib.sha256(f'mx1:{seed}:{epoch}:{step}'.encode()).digest()
    private = random.Random(int.from_bytes(digest, 'big'))
    coefficient = private.betavariate(.2, .2)
    coefficient = max(coefficient, 1-coefficient)
    order = list(range(count))
    private.shuffle(order)
    return coefficient, torch.tensor(order, dtype=torch.int64)


def mix_inputs(images, order, coefficient):
    if not math.isfinite(coefficient) or not 0 <= coefficient <= 1:
        raise ValueError('coefficient must be finite and in [0,1]')
    if images.ndim < 2 or order.dtype != torch.int64 or order.shape != (len(images),):
        raise ValueError('batch permutation shape/dtype mismatch')
    if not torch.equal(order.cpu().sort().values, torch.arange(len(images))):
        raise ValueError('order is not a permutation')
    if coefficient == 1:
        return images
    return coefficient*images + (1-coefficient)*images[order.to(images.device)]


def mixed_weighted_sce(logits, targets, weights, order, coefficient):
    if not math.isfinite(coefficient) or not 0 <= coefficient <= 1:
        raise ValueError('invalid coefficient')
    n = len(logits)
    if targets.shape != (n,) or weights.shape != (n,) or order.shape != (n,):
        raise ValueError('row mismatch')
    if order.dtype != torch.int64 or not torch.equal(order.cpu().sort().values, torch.arange(n)):
        raise ValueError('order is not a permutation')
    if not torch.isfinite(weights).all() or torch.any(weights < 0) or not weights.sum() > 0:
        raise ValueError('weights must have positive finite nonnegative mass')
    order = order.to(targets.device)
    loss_a = symmetric_cross_entropy(logits, targets, alpha=.1, beta=1., log_zero=-4., reduction='none')
    if coefficient == 1:
        return (loss_a*weights).sum()/weights.sum()
    loss_b = symmetric_cross_entropy(logits, targets[order], alpha=.1, beta=1., log_zero=-4., reduction='none')
    left = coefficient*weights
    right = (1-coefficient)*weights[order]
    return (left*loss_a + right*loss_b).sum()/(left+right).sum()
