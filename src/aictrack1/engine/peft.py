"""Checkpoint helpers for single-model parameter-efficient fine-tuning."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not state:
        raise ValueError("model has no trainable parameters")
    return state


def load_trainable_state_dict(
    model: nn.Module, state: Mapping[str, torch.Tensor]
) -> None:
    parameters = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if set(parameters) != set(state):
        missing = sorted(set(parameters) - set(state))
        extra = sorted(set(state) - set(parameters))
        raise ValueError(f"trainable checkpoint mismatch: missing={missing}, extra={extra}")
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = state[name]
            if value.shape != parameter.shape:
                raise ValueError(f"trainable checkpoint shape mismatch for {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def configure_classifier_trainability_from_state(
    classifier: nn.Module,
    state: Mapping[str, torch.Tensor],
) -> bool:
    """Match frozen or jointly trained classifier state before strict PEFT loading."""

    expected = {f"classifier.{name}" for name, _ in classifier.named_parameters()}
    present = {name for name in state if name.startswith("classifier.")}
    if present and present != expected:
        raise ValueError(
            "classifier checkpoint trainable parameter mismatch: "
            f"missing={sorted(expected - present)}, extra={sorted(present - expected)}"
        )
    enabled = bool(present)
    for parameter in classifier.parameters():
        parameter.requires_grad_(enabled)
    return enabled
