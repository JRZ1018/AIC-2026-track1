"""Independent SAM first-order two-backward helper; one caller-owned update.

Source idea: arXiv:2010.01412v3, eq.2/3; no author code copied.
Uses the registered reversible primitive. No optimizer/scaler step or AC update.
"""
from __future__ import annotations

import torch
from aictrack1.methods.sharpness_neighborhood import ReversibleL2Neighborhood

PROTOCOL = dict(schema='aictrack1.same_batch_sam.v1', rho=.05,
                norm='global_l2_trainable', direction='unclipped_first_gradient',
                gradient='first_order_perturbed', input='same_original_strong_batch',
                ac_observation='original_strong_once_per_epoch', teacher=False)


def protocol_state(setting):
    if setting is None:return None
    if setting != PROTOCOL:raise ValueError('SAM1 protocol drift')
    return dict(PROTOCOL)


def verify_protocol_state(saved, setting):
    if saved != protocol_state(setting):raise ValueError('SAM1 checkpoint/config drift')


def same_batch_sam_backward(base_loss, closure, named_parameters, scaler, setting):
    """Leave final scaled .grad at restored base parameters; caller unscales/steps.

    Disabled branch is exactly the historical backward. Enabled branch computes
    scaled first gradients with autograd.grad, whose common positive scale
    cancels in their normalized direction. Only the second backward populates
    .grad. GradScaler.unscale_ is deliberately called once by the caller, after
    this helper returns. AMP overflow uses the original scaler skip/update path;
    no invalid first-gradient direction is ever used for a perturbation.
    """
    protocol_state(setting)
    if setting is None:
        scaler.scale(base_loss).backward()
        return None
    named = list(named_parameters)
    parameters = [p for _,p in named]
    if any(p.grad is not None for p in parameters):
        raise ValueError('SAM1 requires cleared gradients before both backward passes')
    first = torch.autograd.grad(scaler.scale(base_loss), parameters)
    if any(not torch.isfinite(g).all() for g in first):
        if not scaler.is_enabled():
            raise FloatingPointError('SAM1 non-AMP first gradients nonfinite')
        # autograd may return expanded/zero-stride gradients; unscale_ writes in place.
        for p,g in zip(parameters,first):p.grad=g.detach().clone()
        return dict(perturbed_loss=None, first_overflow=True, second_forward=False,
                    second_gradient_overflow=False)
    with ReversibleL2Neighborhood(named, first, rho=setting['rho']):
        second_loss = closure()
        if second_loss.ndim or not torch.isfinite(second_loss):
            raise FloatingPointError('SAM1 perturbed loss must be finite scalar')
        scaler.scale(second_loss).backward()
    if any(p.grad is None for p in parameters):
        raise FloatingPointError('SAM1 second gradients must be complete')
    second_overflow=any(not torch.isfinite(p.grad).all() for p in parameters)
    if second_overflow and not scaler.is_enabled():
        raise FloatingPointError('SAM1 non-AMP second gradients nonfinite')
    return dict(perturbed_loss=second_loss.detach(), first_overflow=False,
                second_forward=True, second_gradient_overflow=second_overflow)
