"""Single-trajectory EMA of trainable parameters; never a second optimizer/model."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Mapping

import torch
from torch import nn


def optimizer_step_completed(*, scaler_enabled: bool, before: float, after: float) -> bool:
    """GradScaler lowers its scale when non-finite gradients skip an optimizer step."""
    return not scaler_enabled or after >= before


class TrainableEMA:
    def __init__(self, model: nn.Module, *, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = float(decay)
        self.parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not self.parameters:
            raise ValueError("EMA requires trainable parameters")
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in self.parameters.items()
        }
        self.updates = 0

    @torch.no_grad()
    def update(self) -> None:
        for name, parameter in self.parameters.items():
            self.shadow[name].lerp_(parameter.detach(), 1.0 - self.decay)
        self.updates += 1

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu().clone() for name, value in self.shadow.items()}

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "updates": self.updates,
            "shadow": self.trainable_state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if float(state["decay"]) != self.decay:
            raise ValueError("EMA decay mismatch on resume")
        shadow = state["shadow"]
        if not isinstance(shadow, dict) or set(shadow) != set(self.parameters):
            raise ValueError("EMA trainable parameter names mismatch on resume")
        if int(state["updates"]) < 0:
            raise ValueError("EMA update count cannot be negative")
        for name, parameter in self.parameters.items():
            value = shadow[name]
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                raise ValueError(f"EMA tensor mismatch: {name}")
            if not torch.isfinite(value).all():
                raise ValueError(f"EMA tensor is non-finite: {name}")
            self.shadow[name] = value.detach().to(parameter.device, parameter.dtype).clone()
        self.updates = int(state["updates"])

    @contextmanager
    def applied(self) -> Iterator[None]:
        """Temporarily evaluate EMA; restore raw weights even if evaluation raises."""
        original = {
            name: parameter.detach().clone()
            for name, parameter in self.parameters.items()
        }
        try:
            with torch.no_grad():
                for name, parameter in self.parameters.items():
                    parameter.copy_(self.shadow[name])
            yield
        finally:
            with torch.no_grad():
                for name, parameter in self.parameters.items():
                    parameter.copy_(original[name])
