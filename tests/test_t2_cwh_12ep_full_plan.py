"""Full-train config for batch 5 item K (12 epochs, min_weight 0.1).

Preregistered expectation: this full run may only be launched if the 12-epoch candidate's
three-seed gate passes. The config must preserve the fixed-split candidate's structure,
learning rate, epoch budget, weighting and seed, and must initialise from the H1 full head
that has seen all 103,218 official-train samples. Inference uses the final-epoch
``final.pt``, never ``best.pt`` (a full-train run has no validation split to select on).
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiment"

FULL = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_12ep_full_seed1_k.yaml"
SEED1 = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_12ep_seed1_k.yaml"
PRIOR_FULL = "p3_stage1_h1init_t2_last4_lr175e6_cwh_8ep_full_seed1_g.yaml"

EXPECTED_CWH = {
    "enabled": True,
    "warmup_epochs": 1,
    "momentum": 0.8,
    "history_mix": 0.5,
    "min_weight": 0.1,
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


def test_full_config_preserves_candidate_definition() -> None:
    full = _load(FULL)
    fixed = _load(SEED1)
    assert full["peft"] == fixed["peft"] == {"mode": "last_n_blocks", "lora_blocks": 4}
    assert full["confidence_weighting"] == fixed["confidence_weighting"] == EXPECTED_CWH
    for key in ("visual_learning_rate", "epochs", "batch_size", "workers", "seed",
                "head_learning_rate", "weight_decay", "gradient_clip_norm"):
        assert full["training"][key] == fixed["training"][key], key
    assert full["training"]["epochs"] == 12


def test_full_config_has_no_validation_knobs_and_no_test_reference() -> None:
    exp = _load(FULL)
    assert "expected_initial_top1" not in exp
    assert "initial_top1_tolerance" not in exp
    assert "test" not in " ".join(str(v) for v in exp.values()).lower()
    assert Path(exp["output_dir"]).name == exp["run_id"]


def test_full_config_does_not_collide_with_the_prior_full_run() -> None:
    new = _load(FULL)
    old = _load(PRIOR_FULL)
    assert new["run_id"] != old["run_id"]
    assert new["output_dir"] != old["output_dir"]
    assert new["training"]["epochs"] == 12
    assert old["training"]["epochs"] == 8
    # same initializer, data and PEFT range; only epochs and the weighting floor differ
    for field in ("dataset_config", "model_config", "manifest", "split", "label_conflicts",
                  "head_checkpoint", "device", "peft"):
        assert new[field] == old[field], field
    assert new["training"]["visual_learning_rate"] == old["training"]["visual_learning_rate"]
    assert new["training"]["seed"] == old["training"]["seed"]
    assert new["confidence_weighting"]["min_weight"] == 0.1
    assert old["confidence_weighting"]["min_weight"] == 0.2
