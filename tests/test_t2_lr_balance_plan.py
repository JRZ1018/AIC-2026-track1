"""Configuration-difference tests for the first batch of seed1 short screening.

Three candidates share every field except the single preregistered variable:

* ``lr16e6``  : visual_learning_rate 1.6e-6
* ``lr165e6`` : visual_learning_rate 1.65e-6
* ``lr175e6_cwh`` : visual_learning_rate 1.75e-6 + classwise history weighting

Everything else (initializer, split, seed, batch, epochs, weight decay, frozen head,
T2 last4 range) must be byte-for-byte identical after removing the run identity and
the one preregistered variable.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiment"

LR16 = "p3_stage1_h1init_t2_last4_lr16e6_6ep_seed1_a.yaml"
LR165 = "p3_stage1_h1init_t2_last4_lr165e6_6ep_seed1_b.yaml"
LR175_CWH = "p3_stage1_h1init_t2_last4_lr175e6_cwh_6ep_seed1_c.yaml"

CONFIGS = (LR16, LR165, LR175_CWH)

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


def _without_variables(name: str) -> dict:
    """Remove run identity plus both preregistered variables."""
    normalized = copy.deepcopy(_load(name))
    normalized.pop("run_id")
    normalized.pop("output_dir")
    normalized.pop("confidence_weighting", None)
    normalized["training"].pop("visual_learning_rate")
    return normalized


def test_first_batch_declares_exactly_three_distinct_learning_rates() -> None:
    rates = {_load(name)["training"]["visual_learning_rate"] for name in CONFIGS}
    assert rates == {0.0000016, 0.00000165, 0.00000175}


def test_first_batch_shared_fields_are_identical() -> None:
    reference = _without_variables(CONFIGS[0])
    for name in CONFIGS[1:]:
        assert _without_variables(name) == reference, f"{name} differs beyond its single variable"


def test_first_batch_is_fixed_split_last4_frozen_head_seed1() -> None:
    for name in CONFIGS:
        exp = _load(name)
        assert exp["execution_enabled"] is True
        assert exp["training_scope"] == "fixed_split_validation"
        assert exp["peft"] == {"mode": "last_n_blocks", "lora_blocks": 4}
        assert exp["device"] == "cuda"
        assert exp["head_checkpoint"].endswith(
            "20260911-p3-h1-linear-adam-e100-seed42/best.pt"
        )
        assert exp["expected_initial_top1"] == 0.686996996
        assert exp["initial_top1_tolerance"] == 0.002
        training = exp["training"]
        assert training["epochs"] == 6
        assert training["batch_size"] == 64
        assert training["workers"] == 4
        assert training["seed"] == 20260910
        assert training["head_learning_rate"] == 0.0
        assert training["weight_decay"] == 0.01
        assert training["gradient_clip_norm"] == 1.0


def test_classwise_history_weighting_only_on_the_cwh_candidate() -> None:
    assert _load(LR16).get("confidence_weighting") is None
    assert _load(LR165).get("confidence_weighting") is None
    assert _load(LR175_CWH)["confidence_weighting"] == EXPECTED_CWH


def test_first_batch_run_identity_is_unique_and_test_free() -> None:
    run_ids = set()
    output_dirs = set()
    for name in CONFIGS:
        exp = _load(name)
        assert "test" not in " ".join(str(value) for value in exp.values()).lower()
        assert Path(exp["output_dir"]).name == exp["run_id"]
        run_ids.add(exp["run_id"])
        output_dirs.add(exp["output_dir"])
    assert len(run_ids) == 3
    assert len(output_dirs) == 3
