"""Independent implementation of LogitClip from Wei et al., ICML 2023.

Paper: https://proceedings.mlr.press/v202/wei23e.html
No source code was copied from the upstream repository, whose audited commit has
no LICENSE file. The training-time transform adds no inference branch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def logit_clip(logits: torch.Tensor, tau: float, p_norm: float = 2.0, eps: float = 1e-12) -> torch.Tensor:
    if tau <= 0:
        raise ValueError("tau must be positive")
    if p_norm <= 0:
        raise ValueError("p_norm must be positive")
    norms = torch.linalg.vector_norm(logits, ord=p_norm, dim=-1, keepdim=True)
    scale = torch.clamp(tau / norms.clamp_min(eps), max=1.0)
    return logits * scale


def logit_clip_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    tau: float,
    p_norm: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    return F.cross_entropy(logit_clip(logits, tau=tau, p_norm=p_norm), targets, reduction=reduction)
