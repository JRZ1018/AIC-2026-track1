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


@dataclass(frozen=True)
class FrozenHeadConfig:
    epochs: int = 30
    batch_size: int = 1024
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    momentum: float = 0.9
    gradient_clip_norm: float = 5.0
    seed: int = 20260910
    optimizer: str = "sgd"
    amp: bool = True

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.optimizer not in {"sgd", "adamw"}:
            raise ValueError("optimizer must be sgd or adamw")


@dataclass(frozen=True)
class FrozenLossConfig:
    kind: str = "cross_entropy"
    gce_q: float = 0.7
    logit_clip_tau: float = 1.5
    logit_clip_p_norm: float = 2.0

    def validate(self) -> None:
        if self.kind not in {"cross_entropy", "gce", "logit_clip_cross_entropy"}:
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
    ) -> None:
        config.validate()
        loss_config.validate()
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.loss_config = loss_config
        if config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
            )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )
        self.amp_enabled = config.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.epoch = 0
        self.best_top1 = float("-inf")
        self.best_state: dict[str, torch.Tensor] | None = None
        self.history: list[dict[str, float | int]] = []

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_config.kind == "cross_entropy":
            return F.cross_entropy(logits, targets)
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
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
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
        state: dict[str, Any] = {
            "config": asdict(self.config),
            "loss_config": asdict(self.loss_config),
            "epoch": self.epoch,
            "best_top1": self.best_top1,
            "model": copy.deepcopy(self.model.state_dict()),
            "best_model": copy.deepcopy(self.best_state),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "scheduler": copy.deepcopy(self.scheduler.state_dict()),
            "scaler": copy.deepcopy(self.scaler.state_dict()),
            "history": copy.deepcopy(self.history),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.random.get_rng_state().clone(),
        }
        if torch.cuda.is_available():
            state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("config") != asdict(self.config):
            raise ValueError("checkpoint training configuration mismatch")
        saved_loss = state.get("loss_config", asdict(FrozenLossConfig()))
        if saved_loss != asdict(self.loss_config):
            raise ValueError("checkpoint loss configuration mismatch")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.epoch = int(state["epoch"])
        self.best_top1 = float(state["best_top1"])
        self.best_state = state["best_model"]
        self.history = state["history"]
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.random.set_rng_state(state["torch_rng_state"])
        if "cuda_rng_state_all" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])

    def save_checkpoint(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        torch.save(self.state_dict(), temporary)
        temporary.replace(output)

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.load_state_dict(state)
