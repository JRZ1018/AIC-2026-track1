"""Frozen LR1 visual recipe; reuse our existing LoRA, never upstream code.

This is a recipe comparison, not proof that low rank alone causes a gain:
the adapter-specific learning rate differs from the full-rank reference.
"""
from __future__ import annotations

import copy
import math

import torch

from aictrack1.models.peft import configure_visual_peft


LR1 = {
    "version": "stage2-lr1-last8-attention-mlp-r8a16-v1",
    "mode": "lora_last_n_attention_mlp",
    "blocks": 8,
    "rank": 8,
    "alpha": 16.0,
    "learning_rate": 1e-4,
    "initialization_seed": 20261005,
}


def validate_recipe(setting):
    if not isinstance(setting, dict) or setting != LR1:
        raise ValueError("unfrozen LR1 recipe")
    return copy.deepcopy(LR1)


def visual_initialization_device(device, setting):
    """Preserve original disabled load; enabled trainer must load on CPU first."""
    if setting is None:
        return device
    validate_recipe(setting)
    return torch.device('cpu')


def configure_low_rank_visual(visual, setting):
    """None retains the exact old last8 path; LR1 init isolates CPU RNG."""
    if setting is None:
        return configure_visual_peft(visual, "last_n_blocks", lora_blocks=8)
    validate_recipe(setting)
    if len(visual.transformer.resblocks) != 12:
        raise ValueError("LR1 requires the official twelve-block visual topology")
    if any(p.device.type != "cpu" for p in visual.parameters()):
        raise ValueError("initialize LR1 on CPU before moving the model to CUDA")
    with torch.random.fork_rng(devices=[]):
        # torch.manual_seed also seeds CUDA generators, even for CPU tensors.
        # Seed ONLY the CPU generator protected by the fork_rng context.
        torch.random.default_generator.manual_seed(LR1["initialization_seed"])
        return configure_visual_peft(
            visual, LR1["mode"], lora_blocks=8,
            lora_rank=8, lora_alpha=16.,
        )


def expected_delta_shapes(width=768, mlp_width=3072):
    shapes = {}
    for block in range(4, 12):
        prefix = f"visual.transformer.resblocks.{block}."
        for module, name, rows, cols in (
            ("attn", "in_proj_weight", width * 3, width),
            ("attn.out_proj", "weight", width, width),
            ("mlp.c_fc", "weight", mlp_width, width),
            ("mlp.c_proj", "weight", width, mlp_width),
        ):
            key = prefix + module + ".parametrizations." + name + ".0."
            shapes[key + "lora_a"] = (8, cols)
            shapes[key + "lora_b"] = (rows, 8)
    return shapes


def validate_delta_state(state, *, width=768, mlp_width=3072):
    shapes = expected_delta_shapes(width, mlp_width)
    if not isinstance(state, dict) or set(state) != set(shapes):
        raise ValueError("LR1 exact trainable tensor set mismatch")
    for name, shape in shapes.items():
        value = state[name]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or not value.is_floating_point() or not torch.isfinite(value).all()):
            raise ValueError("LR1 tensor shape/dtype/finiteness mismatch: " + name)
    return sum(t.numel() for t in state.values())


def exploratory_gate(differences):
    """Predeclared pp gate for NEW candidates, not retrospective adoption.

    Every entry is (point_pp, CI_lower_pp, CI_upper_pp). Conditional bootstrap
    intervals do not include training-seed uncertainty or noisy-label bias.
    """
    if not isinstance(differences, dict) or set(differences) != {"p7", "raw", "ordinary", "tail"}:
        raise ValueError("all predeclared comparisons required")
    for values in differences.values():
        if (len(values) != 3 or not all(math.isfinite(float(x)) for x in values)
                or values[1] > values[2]):
            raise ValueError("invalid finite point/interval in pp")
    p7 = differences["p7"]
    protection = all(
        differences[name][0] >= floor and differences[name][1] >= floor
        for name, floor in (("raw", -.10), ("ordinary", -.10), ("tail", -.25))
    )
    return {
        "exploratory_screen_pass": bool(p7[0] > 0 and p7[1] >= -.10 and protection),
        "original_single_seed_screen_pass": bool(p7[0] >= .10 and p7[1] > 0 and protection),
        "stable_training_gain_proven": False,
        "requires_fresh_full_train": True,
        "test_access_authorized_by_gate_alone": False,
    }
