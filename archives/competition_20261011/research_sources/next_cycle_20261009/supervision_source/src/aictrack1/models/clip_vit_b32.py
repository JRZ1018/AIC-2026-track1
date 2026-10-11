"""Strict local loader for the OpenAI CLIP ViT-B/32 competition backbone."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torchvision.transforms import CenterCrop, Normalize, Resize, ToTensor


OPENAI_CLIP_COMMIT = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CLIPLoadMetadata:
    provider: str
    model_name: str
    upstream_commit: str
    weight_path: str
    weight_sha256: str
    input_resolution: int
    patch_size: int
    feature_dim: int
    preprocessing: dict[str, Any]
    parameter_count: int
    trainable_parameter_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _verify_official_preprocess(preprocess: Any, input_resolution: int) -> None:
    components = getattr(preprocess, "transforms", None)
    if not isinstance(components, list) or len(components) != 5:
        raise ValueError("OpenAI CLIP preprocessing must contain exactly five transforms")
    resize, center_crop, convert_rgb, to_tensor, normalize = components
    if not isinstance(resize, Resize) or resize.size != input_resolution:
        raise ValueError("OpenAI CLIP preprocessing has an unexpected resize")
    if not isinstance(center_crop, CenterCrop) or center_crop.size != (input_resolution, input_resolution):
        raise ValueError("OpenAI CLIP preprocessing has an unexpected center crop")
    if not callable(convert_rgb) or not isinstance(to_tensor, ToTensor):
        raise ValueError("OpenAI CLIP preprocessing must convert RGB and then create a tensor")
    if not isinstance(normalize, Normalize):
        raise ValueError("OpenAI CLIP preprocessing is missing normalization")
    mean = tuple(float(value) for value in normalize.mean)
    std = tuple(float(value) for value in normalize.std)
    if mean != CLIP_MEAN or std != CLIP_STD:
        raise ValueError("OpenAI CLIP preprocessing normalization constants do not match")


def load_clip_vit_b32(
    weight_path: str | Path,
    expected_sha256: str,
    device: str | torch.device,
) -> tuple[torch.nn.Module, Any, CLIPLoadMetadata]:
    """Load exactly the locally supplied OpenAI ViT-B/32 checkpoint.

    A filesystem path is mandatory so the OpenAI package cannot download a model
    implicitly by registry name.
    """
    resolved = Path(weight_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"OpenAI CLIP checkpoint not found: {resolved}")
    actual_sha256 = sha256_file(resolved)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"OpenAI CLIP checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )

    try:
        import clip  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("the official OpenAI clip package is not installed") from exc

    model, preprocess = clip.load(str(resolved), device=device, jit=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    input_resolution = int(model.visual.input_resolution)
    conv = model.visual.conv1
    patch_size = int(conv.kernel_size[0])
    feature_dim = int(model.visual.output_dim)
    if input_resolution != 224 or patch_size != 32 or feature_dim != 512:
        raise ValueError(
            "checkpoint is not OpenAI CLIP ViT-B/32: "
            f"resolution={input_resolution}, patch={patch_size}, feature_dim={feature_dim}"
        )
    _verify_official_preprocess(preprocess, input_resolution)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    metadata = CLIPLoadMetadata(
        provider="OpenAI",
        model_name="ViT-B/32",
        upstream_commit=OPENAI_CLIP_COMMIT,
        weight_path=str(resolved),
        weight_sha256=actual_sha256,
        input_resolution=input_resolution,
        patch_size=patch_size,
        feature_dim=feature_dim,
        preprocessing={
            "resize": 224,
            "center_crop": 224,
            "interpolation": "bicubic",
            "convert_rgb": True,
            "mean": list(CLIP_MEAN),
            "std": list(CLIP_STD),
        },
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_parameter_count,
    )
    return model, preprocess, metadata
