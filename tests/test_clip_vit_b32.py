from __future__ import annotations

import hashlib
import sys
import types

import pytest
import torch
from torch import nn
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor

from aictrack1.models.clip_vit_b32 import load_clip_vit_b32


class _FakeVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_resolution = 224
        self.output_dim = 512
        self.conv1 = nn.Conv2d(3, 4, kernel_size=32, stride=32)


class _FakeCLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _FakeVisual()


def test_strict_local_clip_loader_hashes_freezes_and_identifies_vit_b32(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "ViT-B-32.pt"
    checkpoint.write_bytes(b"official-test-checkpoint")
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    calls = []
    preprocess = Compose(
        [
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    def fake_load(name, device, jit):
        calls.append((name, str(device), jit))
        return _FakeCLIP().to(device), preprocess

    monkeypatch.setitem(sys.modules, "clip", types.SimpleNamespace(load=fake_load))
    model, preprocess, metadata = load_clip_vit_b32(checkpoint, expected, "cpu")
    assert calls == [(str(checkpoint.resolve()), "cpu", False)]
    assert preprocess is not None
    assert metadata.model_name == "ViT-B/32"
    assert metadata.feature_dim == 512
    assert metadata.patch_size == 32
    assert metadata.trainable_parameter_count == 0
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_strict_local_clip_loader_rejects_wrong_hash_before_loading(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "wrong.pt"
    checkpoint.write_bytes(b"wrong")
    monkeypatch.setitem(sys.modules, "clip", types.SimpleNamespace(load=lambda *args: pytest.fail()))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_clip_vit_b32(checkpoint, "0" * 64, "cpu")
