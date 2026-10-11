"""Small CPU-only mathematical and resume checks, no competition image reads."""
from __future__ import annotations
import json
import random
import copy
from portable_training.runtime import SOURCE, import_file


def run():
    import numpy as np
    import torch
    from aictrack1.engine.frozen_head import FrozenHeadConfig, FrozenHeadTrainer
    from aictrack1.models.heads import build_classifier_head
    from aictrack1.methods.sce import symmetric_cross_entropy
    from aictrack1.methods.confidence import ClasswiseHistoryWeighting
    # An import does not execute training. Stub only the head contract import.
    import sys, types
    mod = types.ModuleType('mainline_h1_contract')
    from aictrack1.models.heads import load_classifier_head_checkpoint
    mod.load_new_fold_head = load_classifier_head_checkpoint
    sys.modules['mainline_h1_contract'] = mod
    engine = import_file('portable_scientific_selfcheck', SOURCE / 'scripts/train_n1_peft.py')
    torch.manual_seed(41)
    logits = torch.randn(6, 750, requires_grad=True)
    target = torch.tensor([0, 8, 45, 300, 510, 749])
    losses = engine._base_loss_per_sample(logits, target, kind='sce', q=.7, label_smoothing=0.)
    expected = symmetric_cross_entropy(logits, target, alpha=.1, beta=1., log_zero=-4., reduction='none')
    if not torch.equal(losses, expected): raise AssertionError('Original SCE equality failed')
    weights = torch.tensor([.02, .5, 1., .2, .7, 1.])
    weighted = engine._weighted_base_loss(losses, weights)
    if not torch.equal(weighted, (expected * weights).sum() / weights.sum()):
        raise AssertionError('Original AC weighted loss equality failed')
    weighted.backward()
    if not torch.isfinite(logits.grad).all(): raise AssertionError('Nonfinite synthetic SCE gradients')
    cfg = FrozenHeadConfig(epochs=100, batch_size=4, learning_rate=.01,
        weight_decay=0., momentum=.9, gradient_clip_norm=None, seed=20261008,
        optimizer='adam', scheduler='constant', amp=False)
    trainer = FrozenHeadTrainer(build_classifier_head('linear', 512, 750), cfg,
        device='cpu', num_classes=750)
    x, y = torch.randn(8, 512), torch.arange(8)
    trainer.train_epoch(x, y)
    state = trainer.state_dict()
    restored = FrozenHeadTrainer(build_classifier_head('linear', 512, 750), cfg,
        device='cpu', num_classes=750)
    restored.load_state_dict(copy.deepcopy(state))
    # Both train_epoch calls restore the same serialized RNG first.
    trainer.load_state_dict(copy.deepcopy(state))
    a = trainer.train_epoch(x, y)
    restored.load_state_dict(copy.deepcopy(state))
    b = restored.train_epoch(x, y)
    if a != b or any(not torch.equal(v, restored.model.state_dict()[k]) for k, v in trainer.model.state_dict().items()):
        raise AssertionError('Original H1 optimizer/scheduler/RNG next-step restoration failed')
    print(json.dumps(dict(complete=True, device='cpu', competition_images_read=0,
        training_epochs_reproduced=0, checks=['original_SCE_exact', 'original_AC_weighted_loss_exact',
        'synthetic_finite_gradient', 'H1_optimizer_scheduler_rng_next_step_exact'],
        scope='synthetic functionality only; not a full dataset retraining or score verification')))
    return 0
