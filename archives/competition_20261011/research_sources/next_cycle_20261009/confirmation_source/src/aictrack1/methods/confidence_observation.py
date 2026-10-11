"""N1: same-model train-only weak observations, never pseudo-labels or loss terms."""
from __future__ import annotations

import hashlib
import json
import random
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from aictrack1.data.indexed_images import IndexedImageDataset
from aictrack1.data.transforms import ImageTransform

STRONG = "strong"
WEAK = "weak_center336"
PROTOCOL_VERSION = "aictrack1.n1.observation.v1"


def observation_protocol(view: str, seed: int) -> dict[str, Any]:
    if view not in (STRONG, WEAK):
        raise ValueError("confidence_observation_view must be strong or weak_center336")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("observation seed must be a non-negative integer")
    return {
        "schema": PROTOCOL_VERSION,
        "view": view,
        "training_seed": seed,
        "weak_image_size": 336 if view == WEAK else None,
        "weak_resize_size": 336 if view == WEAK else None,
        "flip_rule": "sha256-json-version-seed-epoch-sample_id-byte0-lsb" if view == WEAK else None,
        "observation_forward": "same-model-pre-step-eval-no_grad-rng-restored" if view == WEAK else "strong-detached-pre-step",
        "targets": "original-observed-labels-only",
    }


def validate_observation_resume(state: Mapping[str, Any], protocol: Mapping[str, Any],
                                *, sources: Mapping[str, str] | None = None) -> None:
    if state.get("confidence_observation_protocol") != dict(protocol):
        raise ValueError("N1 resume observation protocol is missing or differs")
    if sources is not None and state.get("confidence_observation_sources") != dict(sources):
        raise ValueError("N1 resume observation source SHA mismatch")


def weak_flip(seed: int, epoch: int, sample_id: str) -> bool:
    if epoch < 0 or not sample_id:
        raise ValueError("weak observation requires non-negative epoch and stable sample_id")
    payload = json.dumps([PROTOCOL_VERSION, seed, epoch, sample_id], ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return bool(hashlib.sha256(payload).digest()[0] & 1)


class WeakObservationDataset(Dataset[dict[str, object]]):
    """Keep frozen strong __getitem__ unchanged; decode the same train image again.

    Extra decode cost is explicitly part of candidate runtime, not hidden in an
    unverified optimisation of the frozen augmentation/EXIF path.
    """
    def __init__(self, base: IndexedImageDataset, transform: ImageTransform, seed: int) -> None:
        observation_protocol(WEAK, seed)
        if transform.split != "validation" or transform.stochastic:
            raise ValueError("weak transform must be a deterministic center view")
        if transform.config.image_size != 336 or transform.config.resize_size != 336:
            raise ValueError("N1 weak view is frozen at center336/resize336")
        if any(record.source != "train" or record.partition != "train" for record in base.records):
            raise ValueError("weak AC observation accepts only train-partition official train records")
        self.base = base
        self.records = base.records
        self.transform = transform
        self.seed = seed

    @property
    def epoch(self) -> int:
        return self.base.epoch

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, object]:
        batch = self.base[index]
        record = self.records[index]
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = self.base.allow_truncated_images
            with Image.open(record.absolute_path) as opened:
                image = opened.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"failed to load weak train image {record.absolute_path}") from exc
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous
        weak = self.transform(image)
        if weak_flip(self.seed, self.epoch, record.sample_id):
            weak = torch.flip(weak, dims=(-1,))
        batch["confidence_observation_image"] = weak
        return batch


def with_observation_view(base: IndexedImageDataset, transform: ImageTransform,
                          protocol: Mapping[str, Any]) -> IndexedImageDataset | WeakObservationDataset:
    expected = observation_protocol(str(protocol.get("view")), protocol.get("training_seed"))
    if dict(protocol) != expected:
        raise ValueError("unrecognised observation protocol fields")
    if protocol["view"] == STRONG:
        return base  # No extra decode, key, RNG capture or forward in the control.
    return WeakObservationDataset(base, transform, protocol["training_seed"])


@contextmanager
def isolated_eval_no_grad(model: torch.nn.Module, device: torch.device) -> Iterator[None]:
    """Restore every mixed submodule mode and all RNG streams even on failure."""
    modes = [(module, module.training) for module in model.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [] if device.type != "cuda" else [device.index if device.index is not None else torch.cuda.current_device()]
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            yield
    finally:
        for module, mode in modes:
            module.training = mode
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def weak_observation_logits(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    with isolated_eval_no_grad(model, images.device):
        # A no_grad forward must not populate the outer autocast weight cache:
        # reusing its detached casts in the strong forward can disable gradients.
        with torch.autocast(device_type=images.device.type,
                            enabled=torch.is_autocast_enabled(images.device.type),
                            dtype=torch.get_autocast_dtype(images.device.type),
                            cache_enabled=False):
            logits = model(images).detach().float()
        if logits.ndim != 2 or not bool(torch.isfinite(logits).all()):
            raise ValueError("weak observation logits must be finite [batch, classes]")
        return logits


def observation_statistics(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Observe original labels; argmax is only AC history, never a training target."""
    values = logits.detach().float()
    losses = torch.nn.functional.cross_entropy(values, labels, reduction="none")
    probabilities = torch.softmax(values, dim=1).gather(1, labels.view(-1, 1)).squeeze(1)
    return losses.detach(), probabilities.detach(), values.argmax(dim=1).detach()
