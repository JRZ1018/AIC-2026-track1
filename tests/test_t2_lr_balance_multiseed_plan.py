"""Config-difference tests for the multiseed confirmation of the two promoted candidates.

Promoted from the first batch (``docs/54_第一批seed1短筛预注册.md`` section 3):

* C: T2 last4 / visual lr 1.75e-6 / classwise history weighting  (seed1 0.711286724)
* B: T2 last4 / visual lr 1.65e-6 / plain CE                     (seed1 0.706244528)

Each candidate's seed2/seed3 must differ from its seed1 only by ``run_id``,
``output_dir`` and ``training.seed``.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiment"

C_SEED1 = "p3_stage1_h1init_t2_last4_lr175e6_cwh_6ep_seed1_c.yaml"
C_SEED2 = "p3_stage1_h1init_t2_last4_lr175e6_cwh_6ep_seed2_c.yaml"
C_SEED3 = "p3_stage1_h1init_t2_last4_lr175e6_cwh_6ep_seed3_c.yaml"
B_SEED1 = "p3_stage1_h1init_t2_last4_lr165e6_6ep_seed1_b.yaml"
B_SEED2 = "p3_stage1_h1init_t2_last4_lr165e6_6ep_seed2_b.yaml"
B_SEED3 = "p3_stage1_h1init_t2_last4_lr165e6_6ep_seed3_b.yaml"

C_SEEDS = {C_SEED1: 20260910, C_SEED2: 20260911, C_SEED3: 20260912}
B_SEEDS = {B_SEED1: 20260910, B_SEED2: 20260911, B_SEED3: 20260912}

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


def _without_run_identity(name: str) -> dict:
    normalized = copy.deepcopy(_load(name))
    normalized.pop("run_id")
    normalized.pop("output_dir")
    normalized["training"].pop("seed")
    return normalized


def test_c_candidate_multiseed_only_changes_identity_and_seed() -> None:
    reference = _without_run_identity(C_SEED1)
    for name in C_SEED2, C_SEED3:
        assert _without_run_identity(name) == reference
        assert _load(name)["confidence_weighting"] == EXPECTED_CWH


def test_b_candidate_multiseed_only_changes_identity_and_seed() -> None:
    reference = _without_run_identity(B_SEED1)
    for name in B_SEED2, B_SEED3:
        assert _without_run_identity(name) == reference
        assert _load(name).get("confidence_weighting") is None


def test_multiseed_seeds_and_learning_rates_are_exact() -> None:
    for name, seed in {**C_SEEDS, **B_SEEDS}.items():
        assert _load(name)["training"]["seed"] == seed
    for name in C_SEEDS:
        assert _load(name)["training"]["visual_learning_rate"] == 0.00000175
    for name in B_SEEDS:
        assert _load(name)["training"]["visual_learning_rate"] == 0.00000165


def test_multiseed_candidates_are_fixed_split_last4_frozen_head() -> None:
    for name in (*C_SEEDS, *B_SEEDS):
        exp = _load(name)
        assert exp["execution_enabled"] is True
        assert exp["training_scope"] == "fixed_split_validation"
        assert exp["peft"] == {"mode": "last_n_blocks", "lora_blocks": 4}
        assert exp["head_checkpoint"].endswith(
            "20260911-p3-h1-linear-adam-e100-seed42/best.pt"
        )
        assert exp["expected_initial_top1"] == 0.686996996
        training = exp["training"]
        assert training["epochs"] == 6
        assert training["batch_size"] == 64
        assert training["workers"] == 4
        assert training["head_learning_rate"] == 0.0
        assert training["weight_decay"] == 0.01
        assert training["gradient_clip_norm"] == 1.0


def test_multiseed_identity_is_unique_and_test_free() -> None:
    run_ids = set()
    for name in (*C_SEEDS, *B_SEEDS):
        exp = _load(name)
        assert "test" not in " ".join(str(value) for value in exp.values()).lower()
        assert Path(exp["output_dir"]).name == exp["run_id"]
        run_ids.add(exp["run_id"])
    assert len(run_ids) == 6
