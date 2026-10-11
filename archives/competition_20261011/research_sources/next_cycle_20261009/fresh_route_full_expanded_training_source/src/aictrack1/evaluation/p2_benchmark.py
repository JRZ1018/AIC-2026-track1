"""Small known-noise behavior benchmark; not a competition training engine."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from aictrack1.methods.confidence import classwise_rank_noise_scores, select_suspected_noisy
from aictrack1.methods.elr import ELRLoss, ELRState
from aictrack1.methods.logit_clip import logit_clip_cross_entropy

from .noise_detection import evaluate_noise_detection


@dataclass(frozen=True)
class P2BenchmarkConfig:
    seed: int = 20260909
    num_classes: int = 6
    samples_per_class: int = 24
    feature_dim: int = 16
    feature_noise: float = 0.12
    noise_rate: float = 0.3
    steps: int = 30
    learning_rate: float = 0.25
    momentum: float = 0.9
    logit_tau: float = 0.2
    elr_beta: float = 0.7
    elr_lambda: float = 0.5
    min_keep_per_class: int = 2

    def validate(self) -> None:
        if self.num_classes < 2 or self.samples_per_class < 2 or self.feature_dim < self.num_classes:
            raise ValueError("benchmark dimensions are too small")
        if not 0 < self.noise_rate < 1:
            raise ValueError("noise_rate must be in (0, 1)")
        if self.steps < 1 or self.learning_rate <= 0:
            raise ValueError("steps and learning_rate must be positive")
        if self.feature_noise < 0 or self.logit_tau <= 0:
            raise ValueError("feature_noise must be non-negative and logit_tau must be positive")
        if not 0 <= self.momentum < 1 or self.min_keep_per_class < 1:
            raise ValueError("momentum must be in [0, 1) and min_keep_per_class must be positive")


@dataclass(frozen=True)
class KnownNoiseData:
    features: torch.Tensor
    clean_labels: torch.Tensor
    observed_labels: torch.Tensor
    is_noisy: torch.Tensor


def make_known_noise_data(config: P2BenchmarkConfig) -> KnownNoiseData:
    config.validate()
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    clean = torch.arange(config.num_classes).repeat_interleave(config.samples_per_class)
    prototypes = torch.randn(config.num_classes, config.feature_dim, generator=generator)
    prototypes = F.normalize(prototypes, dim=1)
    features = prototypes.index_select(0, clean)
    features = F.normalize(
        features + config.feature_noise * torch.randn(features.shape, generator=generator), dim=1
    )
    observed = clean.clone()
    noisy = torch.zeros_like(clean, dtype=torch.bool)
    per_class_noisy = max(1, round(config.samples_per_class * config.noise_rate))
    for class_id in range(config.num_classes):
        indices = torch.nonzero(clean == class_id, as_tuple=False).flatten()
        selected = indices.index_select(0, torch.randperm(indices.numel(), generator=generator)[:per_class_noisy])
        observed[selected] = (class_id + 1) % config.num_classes
        noisy[selected] = True
    return KnownNoiseData(features, clean, observed, noisy)


class SyntheticNoiseTrainer:
    """Deterministic full-batch trainer used only to verify P2 loss behavior."""

    def __init__(self, data: KnownNoiseData, config: P2BenchmarkConfig, method: str) -> None:
        if method not in {"ce", "logitclip", "elr"}:
            raise ValueError(f"unsupported benchmark method {method!r}")
        self.data = data
        self.config = config
        self.method = method
        with torch.random.fork_rng():
            torch.manual_seed(config.seed + 17)
            self.model = nn.Linear(config.feature_dim, config.num_classes)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=config.learning_rate, momentum=config.momentum
        )
        self.elr_state = (
            ELRState(len(data.clean_labels), config.num_classes, device="cpu", max_bytes=64 * 1024 * 1024)
            if method == "elr"
            else None
        )
        self.elr_loss = (
            ELRLoss(self.elr_state, beta=config.elr_beta, lambda_=config.elr_lambda)
            if self.elr_state is not None
            else None
        )
        self.step_number = 0

    def observed_ce(self) -> torch.Tensor:
        return F.cross_entropy(self.model(self.data.features), self.data.observed_labels)

    def sample_losses(self) -> torch.Tensor:
        with torch.no_grad():
            return F.cross_entropy(
                self.model(self.data.features), self.data.observed_labels, reduction="none"
            )

    def step(self) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(self.data.features)
        if self.method == "ce":
            loss = F.cross_entropy(logits, self.data.observed_labels)
        elif self.method == "logitclip":
            loss = logit_clip_cross_entropy(
                logits, self.data.observed_labels, tau=self.config.logit_tau
            )
        else:
            assert self.elr_loss is not None
            indices = torch.arange(len(self.data.clean_labels))
            loss = self.elr_loss(logits, self.data.observed_labels, indices)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{self.method} produced a non-finite objective")
        loss.backward()
        self.optimizer.step()
        self.step_number += 1
        return float(loss.detach())

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "method": self.method,
            "step_number": self.step_number,
            "model": copy.deepcopy(self.model.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "torch_rng_state": torch.random.get_rng_state().clone(),
        }
        if self.elr_state is not None:
            state["elr"] = self.elr_state.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("method") != self.method:
            raise ValueError("checkpoint method mismatch")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step_number = int(state["step_number"])
        torch.random.set_rng_state(state["torch_rng_state"])
        if self.elr_state is not None:
            if "elr" not in state:
                raise ValueError("ELR checkpoint is missing state")
            self.elr_state.load_state_dict(state["elr"])

    def save_checkpoint(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output)

    def load_checkpoint(self, path: str | Path) -> None:
        self.load_state_dict(torch.load(Path(path), map_location="cpu", weights_only=False))


def run_known_noise_benchmark(config: P2BenchmarkConfig) -> dict[str, Any]:
    data = make_known_noise_data(config)
    methods: dict[str, Any] = {}
    for method in ("ce", "logitclip", "elr"):
        trainer = SyntheticNoiseTrainer(data, config, method)
        initial_ce = float(trainer.observed_ce().detach())
        objectives = [trainer.step() for _ in range(config.steps)]
        losses = trainer.sample_losses()
        scores = classwise_rank_noise_scores(losses, data.observed_labels)
        selected = select_suspected_noisy(
            scores,
            data.observed_labels,
            fraction=config.noise_rate,
            min_keep_per_class=config.min_keep_per_class,
        )
        detection = evaluate_noise_detection(
            data.is_noisy.tolist(),
            selected.tolist(),
            [f"{int(value):04d}" for value in data.observed_labels],
            scores.tolist(),
        )
        methods[method] = {
            "initial_observed_ce": initial_ce,
            "final_observed_ce": float(trainer.observed_ce().detach()),
            "initial_objective": objectives[0],
            "final_objective": objectives[-1],
            "noise_detection": detection,
        }
        if method == "logitclip":
            with torch.no_grad():
                logits = trainer.model(data.features)
                methods[method]["clipped_fraction"] = float(
                    (torch.linalg.vector_norm(logits, dim=1) > 1.0 / config.logit_tau)
                    .float()
                    .mean()
                )
    return {
        "config": asdict(config),
        "sample_count": len(data.clean_labels),
        "realized_noisy_count": int(data.is_noisy.sum()),
        "methods": methods,
        "scope": "synthetic behavior validation only; not a competition accuracy result",
    }
