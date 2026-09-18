"""Config tests for the 2026-09-18 night queue (items V, W, X, Y, Z).

The seed ruler measured the online noise floor at +-0.164, the same order as the recent
single-run gains, so the queue is built to answer one technique question first and then push
only the axis whose historical effect was far larger than that noise.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiment"

BEST = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix75_12ep_full_seed1_o.yaml"
V = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix75_12ep_full_seed1_v.yaml"
W = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix75_20ep_full_seed1_w.yaml"
X = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw02_magmix75_12ep_full_seed1_x.yaml"
Y = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw05_magmix75_16ep_full_seed1_y.yaml"

ALL = (BEST, V, W, X, Y)


def _load(name: str) -> dict:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["experiment"]


def _body(name: str) -> dict:
    config = copy.deepcopy(_load(name))
    config.pop("run_id")
    config.pop("output_dir")
    return config


def test_v_repeats_the_best_config_byte_for_byte() -> None:
    """A same-seed repeat is only a determinism probe if nothing at all differs."""
    assert _body(V) == _body(BEST), "V must be an exact repeat of the 61.0005 configuration"
    assert _load(V)["training"]["seed"] == 20260910


def test_w_changes_only_the_epoch_budget() -> None:
    reference, candidate = _body(BEST), _body(W)
    assert reference["training"]["epochs"] == 12
    assert candidate["training"]["epochs"] == 20
    candidate["training"]["epochs"] = 12
    assert candidate == reference


def test_x_changes_only_the_weighting_floor() -> None:
    reference, candidate = _body(BEST), _body(X)
    assert reference["confidence_weighting"]["min_weight"] == 0.1
    assert candidate["confidence_weighting"]["min_weight"] == 0.02
    candidate["confidence_weighting"]["min_weight"] = 0.1
    assert candidate == reference


def test_y_combines_exactly_two_known_levers() -> None:
    """Y is deliberately the only two-variable member, and both levers are pre-justified."""
    reference, candidate = _body(BEST), _body(Y)
    assert candidate["confidence_weighting"]["min_weight"] == 0.05
    assert candidate["training"]["epochs"] == 16
    candidate["confidence_weighting"]["min_weight"] = 0.1
    candidate["training"]["epochs"] = 12
    assert candidate == reference


def test_every_member_keeps_the_frozen_full_train_setup() -> None:
    for name in ALL:
        experiment = _load(name)
        assert experiment["training_scope"] == "full_train_fixed_epochs", name
        assert experiment["expected_train_count"] == 103218, name
        assert experiment["head_checkpoint"].endswith(
            "20260912-b-final-h1-linear-full-e100-seed1/final.pt"
        ), name
        assert experiment["peft"] == {"mode": "last_n_blocks", "lora_blocks": 4}, name
        assert experiment["output_dir"].endswith(str(experiment["run_id"])), name


def test_z_is_a_seed_lottery_on_the_second_best_recipe() -> None:
    z = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix50_12ep_full_seed3_z.yaml"
    l = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix50_12ep_full_seed1_l.yaml"
    reference, candidate = _body(l), _body(z)
    assert reference["training"]["seed"] == 20260910
    assert candidate["training"]["seed"] == 20260911
    candidate["training"]["seed"] = 20260910
    assert candidate == reference
