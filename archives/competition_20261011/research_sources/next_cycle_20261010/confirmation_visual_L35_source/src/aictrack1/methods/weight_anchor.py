"""Starting-point penalty for current-stage single-model fine-tuning.

The reference consists only of the model's own initial trainable visual weights.
No reference model or external data is loaded during training.
"""

from __future__ import annotations

import torch


def snapshot_trainable_visual(visual: torch.nn.Module) -> tuple[tuple[torch.nn.Parameter, torch.Tensor], ...]:
    """Capture a detached immutable starting point on each parameter's device."""

    pairs = tuple(
        (parameter, parameter.detach().clone())
        for parameter in visual.parameters()
        if parameter.requires_grad
    )
    if not pairs:
        raise ValueError("weight anchoring requires trainable visual parameters")
    return pairs


def starting_point_penalty(
    reference: tuple[tuple[torch.nn.Parameter, torch.Tensor], ...],
    *,
    strength: float,
) -> torch.Tensor:
    """Return 0.5 * strength * sum((current - initial)^2)."""

    if not 0.0 < strength <= 10.0:
        raise ValueError("anchor strength must be in (0, 10]")
    if not reference:
        raise ValueError("anchor reference cannot be empty")
    total = torch.zeros((), dtype=torch.float32, device=reference[0][0].device)
    for parameter, initial in reference:
        if parameter.shape != initial.shape or parameter.device != initial.device:
            raise ValueError("anchor reference shape or device mismatch")
        displacement = parameter.float() - initial.float()
        total = total + displacement.square().sum()
    return 0.5 * float(strength) * total
