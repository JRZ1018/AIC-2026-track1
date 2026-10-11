"""Deterministic linear/cosine probing on frozen CLIP image features."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aictrack1.methods.gce import generalized_cross_entropy
from aictrack1.methods.logit_clip import logit_clip_cross_entropy
from aictrack1.methods.balanced_softmax import balanced_softmax_cross_entropy
from aictrack1.engine.provenance import (
    CheckpointProvenance,
    validate_checkpoint_provenance,
)


@dataclass(frozen=True)
class FrozenHeadConfig:
    epochs: int = 30
    batch_size: int = 1024
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    momentum: float = 0.9
    gradient_clip_norm: float | None = 5.0
    seed: int = 20260910
    optimizer: str = "sgd"
    scheduler: str = "cosine"
    amp: bool = True

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive or null")
        if self.optimizer not in {"sgd", "adam", "adamw"}:
            raise ValueError("optimizer must be sgd, adam, or adamw")
        if self.scheduler not in {"cosine", "constant"}:
            raise ValueError("scheduler must be cosine or constant")


@dataclass(frozen=True)
class FrozenLossConfig:
    kind: str = "cross_entropy"
    gce_q: float = 0.7
    logit_clip_tau: float = 1.5
    logit_clip_p_norm: float = 2.0

    def validate(self) -> None:
        if self.kind not in {"cross_entropy", "gce", "logit_clip_cross_entropy", "balanced_softmax"}:
            raise ValueError(f"unsupported frozen-head loss {self.kind!r}")
        if not 0 < self.gce_q <= 1:
            raise ValueError("gce_q must be in (0, 1]")
        if self.logit_clip_tau <= 0 or self.logit_clip_p_norm <= 0:
            raise ValueError("LogitClip tau and p_norm must be positive")


def _classification_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> dict[str, float]:
    predictions = logits.argmax(dim=1)
    correct = predictions.eq(targets)
    top1 = float(correct.float().mean())
    class_total = torch.bincount(targets, minlength=num_classes)
    class_correct = torch.bincount(targets[correct], minlength=num_classes)
    present = class_total > 0
    macro = float((class_correct[present].float() / class_total[present]).mean())
    return {"top1": top1, "macro_accuracy": macro}


class FrozenHeadTrainer:
    """Epoch-boundary resumable trainer; validation never updates model state."""

    def __init__(
        self,
        model: nn.Module,
        config: FrozenHeadConfig,
        *,
        device: str | torch.device,
        num_classes: int,
        loss_config: FrozenLossConfig = FrozenLossConfig(),
        class_weights: torch.Tensor | None = None,
        class_counts: torch.Tensor | None = None,
        provenance: CheckpointProvenance | None = None,
    ) -> None:
        config.validate()
        loss_config.validate()
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.loss_config = loss_config
        if (loss_config.kind == "balanced_softmax") != (class_counts is not None):
            raise ValueError("balanced_softmax requires class_counts and other losses forbid them")
        if class_counts is not None:
            counts = class_counts.detach().float().cpu()
            if counts.shape != (num_classes,) or not torch.isfinite(counts).all() or bool((counts <= 0).any()):
                raise ValueError("class_counts must be finite positive training counts for every class")
            self.class_counts = counts.to(self.device)
        else:
            self.class_counts = None
        if class_weights is not None:
            weights = class_weights.detach().float().cpu()
            if weights.shape != (num_classes,):
                raise ValueError("class_weights must have shape [num_classes]")
            if not torch.isfinite(weights).all() or bool((weights <= 0).any()):
                raise ValueError("class_weights must be finite and strictly positive")
            if loss_config.kind != "cross_entropy":
                raise ValueError("class_weights currently require cross_entropy")
            self.class_weights = weights.to(self.device)
        else:
            self.class_weights = None
        self.provenance = provenance
        if config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
            )
        elif config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
            )
        else:
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
            )
        if config.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda _: 1.0
            )
        self.amp_enabled = config.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.epoch = 0
        self.best_top1 = float("-inf")
        self.best_state: dict[str, torch.Tensor] | None = None
        self.history: list[dict[str, float | int]] = []

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_config.kind == "cross_entropy":
            return F.cross_entropy(logits, targets, weight=self.class_weights)
        if self.loss_config.kind == "balanced_softmax":
            assert self.class_counts is not None
            return balanced_softmax_cross_entropy(logits, targets, self.class_counts)
        if self.loss_config.kind == "gce":
            return generalized_cross_entropy(logits, targets, q=self.loss_config.gce_q)
        return logit_clip_cross_entropy(
            logits,
            targets,
            tau=self.loss_config.logit_clip_tau,
            p_norm=self.loss_config.logit_clip_p_norm,
        )

    def _loader(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        shuffle: bool,
        epoch: int,
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed + epoch)
        return DataLoader(
            TensorDataset(features, labels),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            generator=generator,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

    def train_epoch(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        self.model.train()
        total_loss = 0.0
        total_count = 0
        for batch_features, batch_labels in self._loader(
            features, labels, shuffle=True, epoch=self.epoch
        ):
            batch_features = batch_features.to(self.device, non_blocking=True)
            batch_labels = batch_labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                loss = self._loss(self.model(batch_features), batch_labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("frozen-head training produced a non-finite loss")
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.config.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += float(loss.detach()) * len(batch_labels)
            total_count += len(batch_labels)
        self.scheduler.step()
        self.epoch += 1
        return total_loss / total_count

    @torch.inference_mode()
    def evaluate(self, features: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        self.model.eval()
        logits = []
        targets = []
        for batch_features, batch_labels in self._loader(
            features, labels, shuffle=False, epoch=0
        ):
            batch_features = batch_features.to(self.device, non_blocking=True)
            # Metrics are always computed from float32 head logits. AMP remains a
            # training optimization, not an evaluation-protocol variable.
            logits.append(self.model(batch_features.float()).float().cpu())
            targets.append(batch_labels)
        return _classification_metrics(torch.cat(logits), torch.cat(targets), self.num_classes)

    def fit(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        validation_features: torch.Tensor,
        validation_labels: torch.Tensor,
    ) -> list[dict[str, float | int]]:
        while self.epoch < self.config.epochs:
            train_loss = self.train_epoch(train_features, train_labels)
            metrics = self.evaluate(validation_features, validation_labels)
            row: dict[str, float | int] = {
                "epoch": self.epoch,
                "train_loss": train_loss,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                **metrics,
            }
            self.history.append(row)
            if metrics["top1"] > self.best_top1:
                self.best_top1 = metrics["top1"]
                self.best_state = copy.deepcopy(self.model.state_dict())
        return self.history

    def state_dict(self) -> dict[str, Any]:
        numpy_name, numpy_keys, numpy_position, numpy_has_gauss, numpy_cached_gaussian = np.random.get_state()
        state: dict[str, Any] = {
            "config": asdict(self.config),
            "loss_config": asdict(self.loss_config),
            "class_weights": (
                None if self.class_weights is None else self.class_weights.detach().cpu().clone()
            ),
            "class_counts": (
                None if self.class_counts is None else self.class_counts.detach().cpu().clone()
            ),
            "epoch": self.epoch,
            "best_top1": self.best_top1,
            "model": copy.deepcopy(self.model.state_dict()),
            "best_model": copy.deepcopy(self.best_state),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "scheduler": copy.deepcopy(self.scheduler.state_dict()),
            "scaler": copy.deepcopy(self.scaler.state_dict()),
            "history": copy.deepcopy(self.history),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": {
                "bit_generator": numpy_name,
                "keys": torch.from_numpy(numpy_keys.astype(np.int64)),
                "position": int(numpy_position),
                "has_gauss": int(numpy_has_gauss),
                "cached_gaussian": float(numpy_cached_gaussian),
            },
            "torch_rng_state": torch.random.get_rng_state().clone(),
        }
        if torch.cuda.is_available():
            state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        if self.provenance is not None:
            state.update(self.provenance.to_dict())
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self.provenance is not None:
            validate_checkpoint_provenance(
                state,
                self.provenance,
                context="frozen-head resume checkpoint",
            )
        saved_config = dict(state.get("config", {}))
        # Checkpoints created before scheduler selection was configurable used
        # CosineAnnealingLR unconditionally.
        saved_config.setdefault("scheduler", "cosine")
        if saved_config != asdict(self.config):
            raise ValueError("checkpoint training configuration mismatch")
        saved_loss = state.get("loss_config", asdict(FrozenLossConfig()))
        if saved_loss != asdict(self.loss_config):
            raise ValueError("checkpoint loss configuration mismatch")
        saved_class_weights = state.get("class_weights")
        expected_class_weights = (
            None if self.class_weights is None else self.class_weights.detach().cpu()
        )
        if (saved_class_weights is None) != (expected_class_weights is None):
            raise ValueError("checkpoint class-weight configuration mismatch")
        if saved_class_weights is not None and not torch.equal(
            saved_class_weights.detach().cpu(), expected_class_weights
        ):
            raise ValueError("checkpoint class weights mismatch")
        saved_class_counts = state.get("class_counts")
        expected_class_counts = (
            None if self.class_counts is None else self.class_counts.detach().cpu()
        )
        if (saved_class_counts is None) != (expected_class_counts is None):
            raise ValueError("checkpoint class counts mismatch")
        if saved_class_counts is not None and not torch.equal(
            saved_class_counts.detach().cpu(), expected_class_counts
        ):
            raise ValueError("checkpoint class counts mismatch")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.epoch = int(state["epoch"])
        self.best_top1 = float(state["best_top1"])
        self.best_state = state["best_model"]
        self.history = state["history"]
        random.setstate(state["python_rng_state"])
        numpy_state = state["numpy_rng_state"]
        if isinstance(numpy_state, dict):
            np.random.set_state(
                (
                    str(numpy_state["bit_generator"]),
                    numpy_state["keys"].detach().cpu().numpy().astype(np.uint32),
                    int(numpy_state["position"]),
                    int(numpy_state["has_gauss"]),
                    float(numpy_state["cached_gaussian"]),
                )
            )
        else:
            # In-memory compatibility for states produced by older versions.
            # Old on-disk pickle checkpoints are intentionally not auto-loaded.
            np.random.set_state(numpy_state)
        # Loading with map_location=self.device moves every tensor, including
        # serialized RNG byte tensors, onto CUDA during GPU resume. The RNG
        # restore APIs require CPU ByteTensors, so move only these bookkeeping
        # tensors back without changing model or optimizer placement.
        torch.random.set_rng_state(state["torch_rng_state"].detach().cpu())
        if "cuda_rng_state_all" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [rng_state.detach().cpu() for rng_state in state["cuda_rng_state_all"]]
            )

    def save_checkpoint(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        torch.save(self.state_dict(), temporary)
        temporary.replace(output)

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(Path(path), map_location=self.device, weights_only=True)
        self.load_state_dict(state)
