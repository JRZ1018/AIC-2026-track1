from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from aictrack1.models.peft import configure_visual_peft, fold_visual_lora


class _Block(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(width, 2)
        self.mlp = nn.Sequential(nn.Linear(width, width * 2), nn.ReLU(), nn.Linear(width * 2, width))
        self.ln_1 = nn.LayerNorm(width)
        self.ln_2 = nn.LayerNorm(width)


class _Visual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 1)
        self.ln_pre = nn.LayerNorm(8)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.Sequential(_Block(), _Block())
        self.ln_post = nn.LayerNorm(8)
        self.proj = nn.Parameter(torch.eye(8))


@pytest.mark.parametrize("mode", ["frozen", "layernorm_projection", "last_block"])
def test_peft_modes_enable_only_expected_parameters(mode: str) -> None:
    visual = _Visual()
    summary = configure_visual_peft(visual, mode)
    names = summary.trainable_parameter_names
    if mode == "frozen":
        assert not names
    elif mode == "layernorm_projection":
        assert names
        assert all(name.startswith(("ln_pre.", "ln_post.")) or name == "proj" for name in names)
    else:
        assert names
        assert all(
            name.startswith("transformer.resblocks.1.")
            or name.startswith("ln_post.")
            or name == "proj"
            for name in names
        )


def test_lora_starts_as_identity_trains_only_low_rank_and_folds() -> None:
    visual = _Visual()
    original = copy.deepcopy(visual.transformer.resblocks[-1].attn.state_dict())
    summary = configure_visual_peft(visual, "lora_last_attention", lora_rank=2, lora_alpha=4)
    assert summary.trainable_visual_parameters == 96
    assert summary.trainable_parameter_names
    assert all("lora_" in name for name in summary.trainable_parameter_names)
    attention = visual.transformer.resblocks[-1].attn
    assert torch.equal(attention.in_proj_weight, original["in_proj_weight"])
    assert torch.equal(attention.out_proj.weight, original["out_proj.weight"])

    before = attention.in_proj_weight.detach().clone()
    with torch.no_grad():
        attention.parametrizations.in_proj_weight[0].lora_b.fill_(0.1)
    changed = attention.in_proj_weight.detach().clone()
    assert not torch.equal(before, changed)
    fold_visual_lora(visual)
    assert torch.equal(attention.in_proj_weight, changed)
    assert not hasattr(attention, "parametrizations") or not hasattr(
        attention.parametrizations, "in_proj_weight"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_lora_parameters_follow_visual_cuda_device() -> None:
    visual = _Visual().cuda()
    configure_visual_peft(visual, "lora_last_attention", lora_rank=2, lora_alpha=4)
    trainable = [parameter for parameter in visual.parameters() if parameter.requires_grad]
    assert trainable
    assert all(parameter.device.type == "cuda" for parameter in trainable)
