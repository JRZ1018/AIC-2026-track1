"""Independent SAM L2 neighborhood primitive, not a trainer or upstream copy.

Idea: Foret et al., arXiv:2010.01412v3 equations 2/3, algorithm 1.
Author repository google-research/sam at dae9904c4cf3a57a304f7b04cecffe371679c702
has Apache-2.0 LICENSE. No author source, data, weights or dependencies copied.
Only explicitly supplied trainable parameters are transiently perturbed.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch

PROTOCOL = 'aictrack1.reversible_l2_neighborhood.v1'


class ReversibleL2Neighborhood:
    """Apply a fixed detached gradient direction and restore by exact copy.

    This object performs no optimizer, scaler, AC or RNG update. The caller must
    keep the same batch/weights for both gradients and checkpoint only at rest.
    It deliberately preserves .grad; the caller owns gradient replacement.
    """

    def __init__(self, named_parameters: Iterable[tuple[str, torch.Tensor]],
                 gradients: Iterable[torch.Tensor], *, rho: float = .05,
                 direction: int = 1) -> None:
        self.parameters = list(named_parameters)
        self.gradients = list(gradients)
        self.rho = float(rho)
        self.direction = direction
        self.active = False
        self._saved: list[torch.Tensor] = []
        if not math.isfinite(self.rho) or self.rho < 0 or direction not in (-1, 1):
            raise ValueError('finite nonnegative rho and direction +/-1 required')
        names = [name for name, _ in self.parameters]
        if not names or len(names) != len(set(names)):
            raise ValueError('nonempty unique parameter names required')
        if len({id(p) for _, p in self.parameters}) != len(self.parameters):
            raise ValueError('aliased parameter objects are forbidden')
        if len(self.gradients) != len(self.parameters):
            raise ValueError('one gradient per parameter required')
        for (_, p), g in zip(self.parameters, self.gradients):
            if not p.requires_grad or not p.is_floating_point() or not torch.isfinite(p).all():
                raise ValueError('finite floating trainable parameter required')
            if not isinstance(g, torch.Tensor) or g.shape != p.shape or g.device != p.device:
                raise ValueError('gradient shape/device mismatch')
            if not g.is_floating_point() or not torch.isfinite(g).all():
                raise ValueError('finite floating gradient required')
        # Float64 norm reduction avoids FP16/FP32 squared-norm overflow.
        norms = [float(torch.linalg.vector_norm(g.detach(), dtype=torch.float64))
                 for g in self.gradients]
        self.gradient_norm = math.hypot(*norms)
        if not math.isfinite(self.gradient_norm):
            raise ValueError('nonfinite global gradient norm')

    def state_dict(self) -> dict:
        if self.active:
            raise RuntimeError('cannot checkpoint a transient perturbation')
        return dict(protocol=PROTOCOL, rho=self.rho, direction=self.direction,
                    parameter_names=[n for n, _ in self.parameters], active=False)

    def load_state_dict(self, state: dict) -> None:
        if self.active or state != self.state_dict():
            raise ValueError('neighborhood protocol/config/parameter drift')

    def __enter__(self) -> 'ReversibleL2Neighborhood':
        if self.active or self._saved:
            raise RuntimeError('nested perturbation forbidden')
        self.active = True
        try:
            if self.rho and self.gradient_norm:
                self._saved = [p.detach().clone() for _, p in self.parameters]
                with torch.no_grad():
                    for (_, p), g in zip(self.parameters, self.gradients):
                        p.add_(g.detach().to(p.dtype), alpha=self.direction*self.rho/self.gradient_norm)
                        if not torch.isfinite(p).all():
                            raise ValueError('perturbation produced nonfinite parameter')
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        try:
            with torch.no_grad():
                for (_, p), saved in zip(self.parameters, self._saved):
                    p.copy_(saved)
        finally:
            self._saved = []
            self.active = False
        return False
