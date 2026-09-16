"""Config-difference tests for batch 3: min_weight=0.5 and the 8-epoch extension.

Both candidates must differ from the passing 6-epoch / min_weight=0.2 config
(``p3_stage1_h1init_t2_last4_lr175e6_cwh_6ep_seed1_c.yaml``) by exactly their single
preregistered variable.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiment"

REFERENCE = "p3_stage1_h1init_t2_last4_lr175e6_cwh_6ep_seed1_c.yaml"
MW50 = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw50_6ep_seed1_f.yaml"
EP8 = "p3_stage1_h1init_t2_last4_lr175e6_cwh_8ep_seed1_g.yaml"

CWH_02 = {
    "enabled": True,
    "warmup_epochs": 1,
    "momentum": 0.8,
    "history_mix": 0.5,
    "min_weight": 0.2,
}


def _load(name: str) -> dict:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["experiment"]


def _drop_identity(config: dict) -> dict:
    normalized = copy.deepcopy(config)
    normalized.pop("run_id")
    normalized.pop("output_dir")
    return normalized


def test_reference_is_the_passing_six_epoch_floor_02_config() -> None:
    exp = _load(REFERENCE)
    assert exp["confidence_weighting"] == CWH_02
    assert exp["training"]["epochs"] == 6
    assert exp["training"]["visual_learning_rate"] == 0.00000175


def test_mw50_candidate_changes_only_the_floor() -> None:
    reference = _drop_identity(_load(REFERENCE))
    candidate = _drop_identity(_load(MW50))
    assert candidate["confidence_weighting"] == {**CWH_02, "min_weight": 0.5}
    candidate["confidence_weighting"] = CWH_02
    assert candidate == reference


def test_eight_epoch_candidate_changes_only_the_epoch_budget() -> None:
    reference = _drop_identity(_load(REFERENCE))
    candidate = _drop_identity(_load(EP8))
    assert candidate["training"]["epochs"] == 8
    assert candidate["confidence_weighting"] == CWH_02
    candidate["training"]["epochs"] = 6
    assert candidate == reference


def test_batch3_shared_protocol() -> None:
    for name in (MW50, EP8):
        exp = _load(name)
        assert exp["execution_enabled"] is True
        assert exp["training_scope"] == "fixed_split_validation"
        assert exp["peft"] == {"mode": "last_n_blocks", "lora_blocks": 4}
        assert exp["head_checkpoint"].endswith(
            "20260911-p3-h1-linear-adam-e100-seed42/best.pt"
        )
        assert exp["expected_initial_top1"] == 0.686996996
        assert exp["initial_top1_tolerance"] == 0.002
        training = exp["training"]
        assert training["batch_size"] == 64
        assert training["workers"] == 4
        assert training["seed"] == 20260910
        assert training["visual_learning_rate"] == 0.00000175
        assert training["head_learning_rate"] == 0.0
        assert training["weight_decay"] == 0.01
        assert training["gradient_clip_norm"] == 1.0


def test_batch3_identity_is_unique_and_test_free() -> None:
    run_ids = set()
    for name in (MW50, EP8):
        exp = _load(name)
        assert "test" not in " ".join(str(value) for value in exp.values()).lower()
        assert Path(exp["output_dir"]).name == exp["run_id"]
        run_ids.add(exp["run_id"])
    assert len(run_ids) == 2
