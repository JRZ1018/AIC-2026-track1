"""Config tests for batch 6 item N: the soft-weighting mechanism switched off.

Every experiment on this route has carried classwise-history soft weighting, so the
mechanism's own contribution has never been measured. Item N removes it entirely (the
trainer only builds the weighting state when `confidence_weighting` is present), which
makes the mechanism itself the single variable.

N must therefore differ from the frozen reference by the absence of that block and
nothing else.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiment"

SCREEN = "p3_stage1_h1init_t2_last4_lr175e6_nocwh_12ep_seed1_n.yaml"
FULL = "p3_stage1_h1init_t2_last4_lr175e6_nocwh_12ep_full_seed1_n.yaml"
REFERENCE = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_12ep_seed1_k.yaml"
FULL_REFERENCE = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_12ep_full_seed1_k.yaml"


def _load(name: str) -> dict:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["experiment"]


def _drop_identity(config: dict) -> dict:
    normalized = copy.deepcopy(config)
    normalized.pop("run_id")
    normalized.pop("output_dir")
    return normalized


def test_screen_removes_only_the_weighting_block() -> None:
    reference = _drop_identity(_load(REFERENCE))
    candidate = _drop_identity(_load(SCREEN))

    assert "confidence_weighting" in reference
    assert "confidence_weighting" not in candidate
    reference.pop("confidence_weighting")
    assert reference == candidate


def test_full_removes_only_the_weighting_block() -> None:
    reference = _drop_identity(_load(FULL_REFERENCE))
    candidate = _drop_identity(_load(FULL))

    assert "confidence_weighting" in reference
    assert "confidence_weighting" not in candidate
    reference.pop("confidence_weighting")
    assert reference == candidate


def test_screen_keeps_the_frozen_head_and_budget() -> None:
    experiment = _load(SCREEN)
    assert experiment["training_scope"] == "fixed_split_validation"
    assert experiment["head_checkpoint"].endswith(
        "20260911-p3-h1-linear-adam-e100-seed42/best.pt"
    )
    assert experiment["training"]["epochs"] == 12
    assert experiment["training"]["seed"] == 20260910
    assert experiment["training"]["visual_learning_rate"] == 0.00000175
    assert experiment["peft"] == {"mode": "last_n_blocks", "lora_blocks": 4}
    # The initial-parity guard runs before any weighting is applied, so it stays valid.
    assert experiment["expected_initial_top1"] == 0.686996996


def test_full_covers_the_whole_split_with_the_scored_head() -> None:
    experiment = _load(FULL)
    assert experiment["training_scope"] == "full_train_fixed_epochs"
    assert experiment["expected_train_count"] == 103218
    assert experiment["head_checkpoint"].endswith(
        "20260912-b-final-h1-linear-full-e100-seed1/final.pt"
    )
    assert "expected_initial_top1" not in experiment
    assert experiment["output_dir"].endswith(str(experiment["run_id"]))


def test_no_other_config_on_the_route_drops_the_weighting_block() -> None:
    """Guard against a second, unintended mechanism-off run appearing later."""
    offenders = []
    for path in sorted(CONFIG_DIR.glob("p3_stage1_h1init_t2_last4_*_cwh_*.yaml")):
        with path.open("r", encoding="utf-8") as handle:
            if "confidence_weighting" not in yaml.safe_load(handle)["experiment"]:
                offenders.append(path.name)
    assert offenders == []
