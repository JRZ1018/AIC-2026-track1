"""Preregistered full-train config for the promoted C candidate.

The C candidate (T2 last4 / visual lr 1.75e-6 / classwise history weighting) passed the
three-seed gate, so exactly one enabled full-train config may exist for it. This module
locks that config's shape: full official-train scope, the leakage-free H1 full initializer,
and the identical structure, learning rate, epochs, weighting and seed as the fixed-split
candidate.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiment"

FULL = "p3_stage1_h1init_t2_last4_lr175e6_cwh_6ep_full_seed1_c.yaml"
SEED1 = "p3_stage1_h1init_t2_last4_lr175e6_cwh_6ep_seed1_c.yaml"
SEED1_V2 = "p3_stage1_h1init_t2_last4_lr175e6_cwh_6ep_seed1_v2_c.yaml"

EXPECTED_CWH = {
    "enabled": True,
    "warmup_epochs": 1,
    "momentum": 0.8,
    "history_mix": 0.5,
    "min_weight": 0.2,
}


def _load(name: str) -> dict:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["experiment"]


def test_full_config_is_enabled_full_scope_with_expected_train_count() -> None:
    exp = _load(FULL)
    assert exp["execution_enabled"] is True
    assert exp["training_scope"] == "full_train_fixed_epochs"
    assert exp["expected_train_count"] == 103218


def test_full_config_uses_the_leak_free_h1_full_initializer() -> None:
    exp = _load(FULL)
    assert exp["head_checkpoint"].endswith(
        "20260912-b-final-h1-linear-full-e100-seed1/final.pt"
    )


def test_full_config_preserves_candidate_structure_lr_epochs_weighting_and_seed() -> None:
    full = _load(FULL)
    fixed = _load(SEED1)
    assert full["peft"] == fixed["peft"] == {"mode": "last_n_blocks", "lora_blocks": 4}
    assert full["confidence_weighting"] == fixed["confidence_weighting"] == EXPECTED_CWH
    assert full["training"]["visual_learning_rate"] == fixed["training"]["visual_learning_rate"]
    assert full["training"]["epochs"] == fixed["training"]["epochs"] == 6
    assert full["training"]["batch_size"] == fixed["training"]["batch_size"] == 64
    assert full["training"]["workers"] == fixed["training"]["workers"] == 4
    assert full["training"]["seed"] == fixed["training"]["seed"] == 20260910
    assert full["training"]["head_learning_rate"] == 0.0
    assert full["training"]["weight_decay"] == fixed["training"]["weight_decay"] == 0.01
    assert full["training"]["gradient_clip_norm"] == fixed["training"]["gradient_clip_norm"] == 1.0


def test_full_config_has_no_validation_knobs_and_no_test_reference() -> None:
    exp = _load(FULL)
    # A full-train run has no validation split to select on.
    assert "expected_initial_top1" not in exp
    assert "initial_top1_tolerance" not in exp
    assert "test" not in " ".join(str(value) for value in exp.values()).lower()
    assert Path(exp["output_dir"]).name == exp["run_id"]


def test_seed1_v2_reruns_the_same_candidate_config_under_a_new_identity() -> None:
    """seed1-v2 exists only to give seed1 the same training commit as seed2/seed3."""
    reference = copy.deepcopy(_load(SEED1))
    reference.pop("run_id")
    reference.pop("output_dir")
    v2 = copy.deepcopy(_load(SEED1_V2))
    v2.pop("run_id")
    v2.pop("output_dir")
    assert v2 == reference
    assert _load(SEED1_V2)["training"]["seed"] == 20260910
    assert _load(SEED1_V2)["confidence_weighting"] == EXPECTED_CWH
