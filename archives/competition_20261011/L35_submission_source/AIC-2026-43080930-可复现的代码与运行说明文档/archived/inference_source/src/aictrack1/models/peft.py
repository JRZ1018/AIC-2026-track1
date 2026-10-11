"""Single-model PEFT controls for the OpenAI CLIP ViT-B/32 visual encoder."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from torch import nn


@dataclass(frozen=True)
class PEFTSummary:
    mode: str
    total_visual_parameters: int
    trainable_visual_parameters: int
    trainable_parameter_names: tuple[str, ...]


class LowRankWeightDelta(nn.Module):
    """Add a trainable low-rank delta while leaving the original weight frozen."""

    def __init__(self, weight: torch.Tensor, *, rank: int, alpha: float) -> None:
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("LoRA weight must be a matrix")
        if rank < 1 or rank > min(weight.shape):
            raise ValueError("LoRA rank is incompatible with the weight shape")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        self.lora_a = nn.Parameter(
            torch.empty(rank, weight.shape[1], device=weight.device, dtype=weight.dtype)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(weight.shape[0], rank, device=weight.device, dtype=weight.dtype)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        return original + (self.lora_b @ self.lora_a).to(original.dtype) * self.scale


def _last_visual_block(visual: nn.Module) -> tuple[int, nn.Module]:
    blocks = visual.transformer.resblocks
    if len(blocks) < 1:
        raise ValueError("CLIP visual transformer has no residual blocks")
    index = len(blocks) - 1
    return index, blocks[index]


def _register_lora(module: nn.Module, name: str, *, rank: int, alpha: float) -> None:
    weight = getattr(module, name)
    parametrize.register_parametrization(
        module,
        name,
        LowRankWeightDelta(weight, rank=rank, alpha=alpha),
    )


def configure_visual_peft(
    visual: nn.Module,
    mode: str,
    *,
    lora_rank: int = 4,
    lora_alpha: float = 8.0,
    lora_blocks: int = 1,
) -> PEFTSummary:
    """Enable exactly one audited visual adaptation strategy."""

    allowed = {
        "frozen",
        "layernorm_projection",
        "last_block",
        "last_n_blocks",
        "full_visual",
        "lora_last_attention",
        "lora_last_n_attention",
        "lora_last_n_attention_mlp",
    }
    if mode not in allowed:
        raise ValueError(f"unsupported PEFT mode {mode!r}")
    for parameter in visual.parameters():
        parameter.requires_grad_(False)

    last_index, last_block = _last_visual_block(visual)
    if mode == "full_visual":
        # Image tower only: include patch/CLS/position, every residual block,
        # layer norms and output projection. Classifier trainability is separate.
        for parameter in visual.parameters():
            parameter.requires_grad_(True)
    elif mode == "layernorm_projection":
        for name, parameter in visual.named_parameters():
            if name.startswith("ln_pre.") or name.startswith("ln_post.") or name == "proj":
                parameter.requires_grad_(True)
    elif mode == "last_block":
        prefix = f"transformer.resblocks.{last_index}."
        for name, parameter in visual.named_parameters():
            if name.startswith(prefix) or name.startswith("ln_post.") or name == "proj":
                parameter.requires_grad_(True)
    elif mode == "last_n_blocks":
        # Only the number of fully unfrozen residual blocks changes; ln_post and
        # proj stay frozen so the comparison against `last_block` isolates block
        # count instead of also varying the output projection.
        block_count = int(lora_blocks)
        if block_count < 1 or block_count > len(visual.transformer.resblocks):
            raise ValueError("block count must be in [1, number of visual blocks]")
        prefixes = tuple(
            f"transformer.resblocks.{index}."
            for index in range(len(visual.transformer.resblocks) - block_count, len(visual.transformer.resblocks))
        )
        for name, parameter in visual.named_parameters():
            if name.startswith(prefixes):
                parameter.requires_grad_(True)
    elif mode in {
        "lora_last_attention",
        "lora_last_n_attention",
        "lora_last_n_attention_mlp",
    }:
        block_count = 1 if mode == "lora_last_attention" else int(lora_blocks)
        if block_count < 1 or block_count > len(visual.transformer.resblocks):
            raise ValueError("lora_blocks must be in [1, number of visual blocks]")
        for block in visual.transformer.resblocks[-block_count:]:
            attention = block.attn
            if not isinstance(attention, nn.MultiheadAttention):
                raise TypeError("CLIP visual attention is not torch.nn.MultiheadAttention")
            _register_lora(attention, "in_proj_weight", rank=lora_rank, alpha=lora_alpha)
            _register_lora(attention.out_proj, "weight", rank=lora_rank, alpha=lora_alpha)
            if mode == "lora_last_n_attention_mlp":
                for name in ("c_fc", "c_proj"):
                    layer = getattr(block.mlp, name, None)
                    if not isinstance(layer, nn.Linear):
                        raise TypeError(f"CLIP visual MLP {name} is not torch.nn.Linear")
                    _register_lora(layer, "weight", rank=lora_rank, alpha=lora_alpha)

    names = tuple(name for name, parameter in visual.named_parameters() if parameter.requires_grad)
    if mode != "frozen" and not names:
        raise RuntimeError(f"PEFT mode {mode} enabled no visual parameters")
    if mode in {
        "lora_last_attention",
        "lora_last_n_attention",
        "lora_last_n_attention_mlp",
    } and any(
        "lora_" not in name for name in names
    ):
        raise RuntimeError("LoRA mode unexpectedly enabled a non-LoRA visual parameter")
    return PEFTSummary(
        mode=mode,
        total_visual_parameters=sum(parameter.numel() for parameter in visual.parameters()),
        trainable_visual_parameters=sum(
            parameter.numel() for parameter in visual.parameters() if parameter.requires_grad
        ),
        trainable_parameter_names=names,
    )


def fold_visual_lora(visual: nn.Module) -> None:
    """Fold registered LoRA deltas into the single visual model for inference."""

    for block in visual.transformer.resblocks:
        attention = block.attn
        if parametrize.is_parametrized(attention, "in_proj_weight"):
            parametrize.remove_parametrizations(
                attention, "in_proj_weight", leave_parametrized=True
            )
        if parametrize.is_parametrized(attention.out_proj, "weight"):
            parametrize.remove_parametrizations(
                attention.out_proj, "weight", leave_parametrized=True
            )
        for name in ("c_fc", "c_proj"):
            layer = getattr(block.mlp, name, None)
            if layer is not None and parametrize.is_parametrized(layer, "weight"):
                parametrize.remove_parametrizations(layer, "weight", leave_parametrized=True)


class PEFTImageClassifier(nn.Module):
    def __init__(
        self,
        visual: nn.Module,
        classifier: nn.Module,
        *,
        normalize_features: bool = False,
    ) -> None:
        super().__init__()
        self.visual = visual
        self.classifier = classifier
        self.normalize_features = normalize_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.visual(images).float()
        if self.normalize_features:
            features = F.normalize(features, dim=-1)
        return self.classifier(features)
