"""Memory-guarded single-model Early-Learning Regularization.

Paper: https://proceedings.neurips.cc/paper/2020/hash/ea89621bee7c88b2c5be6681c8ef4906-Abstract.html
This is an independent, minimal implementation of basic ELR only; ELR+ is
deliberately excluded because the project requires a conservative single-model path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def estimate_elr_state_bytes(num_samples: int, num_classes: int, dtype: torch.dtype) -> int:
    if num_samples <= 0 or num_classes <= 0:
        raise ValueError("num_samples and num_classes must be positive")
    return num_samples * num_classes * torch.empty((), dtype=dtype).element_size()


@dataclass
class ELRState:
    num_samples: int
    num_classes: int
    dtype: torch.dtype = torch.float32
    device: torch.device | str = "cpu"
    max_bytes: int | None = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        required = estimate_elr_state_bytes(self.num_samples, self.num_classes, self.dtype)
        if self.max_bytes is not None and required > self.max_bytes:
            raise MemoryError(f"ELR state requires {required} bytes, exceeding limit {self.max_bytes}")
        self.device = torch.device(self.device)
        self.target = torch.zeros((self.num_samples, self.num_classes), dtype=self.dtype, device=self.device)

    @property
    def nbytes(self) -> int:
        return self.target.numel() * self.target.element_size()

    def state_dict(self) -> dict[str, object]:
        return {
            "num_samples": self.num_samples,
            "num_classes": self.num_classes,
            "dtype": str(self.dtype),
            "target": self.target.detach().cpu(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state["num_samples"]) != self.num_samples or int(state["num_classes"]) != self.num_classes:
            raise ValueError("ELR state shape metadata mismatch")
        target = state["target"]
        if not isinstance(target, torch.Tensor) or tuple(target.shape) != (self.num_samples, self.num_classes):
            raise ValueError("invalid ELR target tensor")
        self.target.copy_(target.to(device=self.device, dtype=self.dtype))


class ELRLoss(nn.Module):
    def __init__(self, state: ELRState, *, beta: float = 0.7, lambda_: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        if not 0 <= beta < 1:
            raise ValueError("beta must be in [0, 1)")
        if lambda_ < 0:
            raise ValueError("lambda_ must be non-negative")
        self.state = state
        self.beta = beta
        self.lambda_ = lambda_
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, sample_indices: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] != self.state.num_classes:
            raise ValueError("logits shape does not match ELR state")
        if targets.shape != sample_indices.shape or targets.ndim != 1:
            raise ValueError("targets and sample_indices must be matching 1D tensors")
        if torch.unique(sample_indices).numel() != sample_indices.numel():
            raise ValueError("sample_indices must be unique within a batch")
        indices = sample_indices.detach().to(device=self.state.device, dtype=torch.long)
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= self.state.num_samples):
            raise IndexError("sample index outside ELR state")
        probs = F.softmax(logits.float(), dim=1).clamp(self.eps, 1.0 - self.eps)
        probs = probs / probs.sum(dim=1, keepdim=True)
        previous = self.state.target.index_select(0, indices).to(device=logits.device, dtype=probs.dtype)
        updated = self.beta * previous + (1.0 - self.beta) * probs.detach()
        self.state.target.index_copy_(0, indices, updated.to(device=self.state.device, dtype=self.state.dtype))
        agreement = (updated * probs).sum(dim=1)
        regularizer = torch.log((1.0 - agreement).clamp_min(self.eps)).mean()
        return F.cross_entropy(logits, targets) + self.lambda_ * regularizer
