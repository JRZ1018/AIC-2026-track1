"""Full-train config for the 8-epoch cwh candidate.

Preregistered expectation: this full run may only be launched if the 8-epoch candidate's
three-seed gate passes. The config must preserve the fixed-split candidate's structure,
learning rate, epoch budget, weighting and seed, and must initialise from the H1 full head
that has seen all 103,218 official-train samples.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiment"

FULL = "p3_stage1_h1init_t2_last4_lr175e6_cwh_8ep_full_seed1_g.yaml"
SEED1 = "p3_stage1_h1init_t2_last4_lr175e6_cwh_8ep_seed1_g.yaml"

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
    assert _load(FULL)["head_checkpoint"].endswith(
        "20260912-b-final-h1-linear-full-e100-seed1/final.pt"
    )


def test_full_config_preserves_candidate_structure_lr_epochs_weighting_and_seed() -> None:
    full = _load(FULL)
    fixed = _load(SEED1)
    assert full["peft"] == fixed["peft"] == {"mode": "last_n_blocks", "lora_blocks": 4}
    assert full["confidence_weighting"] == fixed["confidence_weighting"] == EXPECTED_CWH
    assert full["training"]["visual_learning_rate"] == fixed["training"]["visual_learning_rate"]
    assert full["training"]["epochs"] == fixed["training"]["epochs"] == 8
    assert full["training"]["batch_size"] == fixed["training"]["batch_size"] == 64
    assert full["training"]["workers"] == fixed["training"]["workers"] == 4
    assert full["training"]["seed"] == fixed["training"]["seed"] == 20260910
    assert full["training"]["head_learning_rate"] == 0.0
    assert full["training"]["weight_decay"] == fixed["training"]["weight_decay"] == 0.01
    assert full["training"]["gradient_clip_norm"] == fixed["training"]["gradient_clip_norm"] == 1.0


def test_full_config_has_no_validation_knobs_and_no_test_reference() -> None:
    exp = _load(FULL)
    assert "expected_initial_top1" not in exp
    assert "initial_top1_tolerance" not in exp
    assert "test" not in " ".join(str(v) for v in exp.values()).lower()
    assert Path(exp["output_dir"]).name == exp["run_id"]


def test_candidate_identity_differs_from_the_six_epoch_full_config() -> None:
    """The 8-epoch full run must not collide with the existing 6-epoch cwh full run."""
    new = _load(FULL)
    old = _load("p3_stage1_h1init_t2_last4_lr175e6_cwh_6ep_full_seed1_c.yaml")
    assert new["run_id"] != old["run_id"]
    assert new["output_dir"] != old["output_dir"]
    assert new["training"]["epochs"] == 8
    assert old["training"]["epochs"] == 6
    unchanged = ("dataset_config", "model_config", "manifest", "split", "label_conflicts",
                 "head_checkpoint", "device", "peft", "confidence_weighting")
    for field in unchanged:
        assert new[field] == old[field], field
    assert new["training"]["visual_learning_rate"] == old["training"]["visual_learning_rate"]
    assert new["training"]["seed"] == old["training"]["seed"]
