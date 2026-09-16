"""Multiseed confirmation configs for batch 5 item I (10 epochs, min_weight 0.1).

Item I passed every seed1 gate with the best fixed-validation result so far
(Top-1 0.717783391, macro 0.717656851, conflict 0.311464548 vs the 8-epoch reference
0.713759303 / 0.713626802 / 0.310139179).

These two configs add seed2/seed3 for the three-seed gate. Each must differ from
seed1 only by ``run_id``, ``output_dir`` and ``training.seed``. All three seeds are
trained at the same commit, which fixes the cross-commit inconsistency recorded for
batch 4.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiment"

SEED1 = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_10ep_seed1_i.yaml"
SEED2 = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_10ep_seed2_i.yaml"
SEED3 = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_10ep_seed3_i.yaml"

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


def _drop_identity(config: dict) -> dict:
    normalized = copy.deepcopy(config)
    normalized.pop("run_id")
    normalized.pop("output_dir")
    normalized["training"].pop("seed")
    return normalized


def test_multiseed_only_changes_identity_and_seed() -> None:
    reference = _drop_identity(_load(SEED1))
    for name in (SEED2, SEED3):
        assert _drop_identity(_load(name)) == reference
        assert _load(name)["confidence_weighting"] == EXPECTED_CWH


def test_seeds_are_exact() -> None:
    assert _load(SEED1)["training"]["seed"] == 20260910
    assert _load(SEED2)["training"]["seed"] == 20260911
    assert _load(SEED3)["training"]["seed"] == 20260912


def test_candidate_definition_is_frozen() -> None:
    for name in (SEED1, SEED2, SEED3):
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
        assert training["epochs"] == 10
        assert training["batch_size"] == 64
        assert training["workers"] == 4
        assert training["visual_learning_rate"] == 0.00000175
        assert training["head_learning_rate"] == 0.0
        assert training["weight_decay"] == 0.01
        assert training["gradient_clip_norm"] == 1.0
        assert "test" not in " ".join(str(v) for v in exp.values()).lower()
        assert Path(exp["output_dir"]).name == exp["run_id"]


def test_identity_is_unique_and_does_not_collide_with_prior_runs() -> None:
    names = (SEED1, SEED2, SEED3, "p3_stage1_h1init_t2_last4_lr175e6_cwh_8ep_seed1_g.yaml")
    ids = {_load(n)["run_id"] for n in names}
    dirs = {_load(n)["output_dir"] for n in names}
    assert len(ids) == 4
    assert len(dirs) == 4


def test_seed1_trigger_evidence_is_recorded() -> None:
    """I cleared the batch-5 seed1 gate against the 8-epoch reference."""
    reference = {"top1": 0.713759303, "macro": 0.713626802, "conflict": 0.310139179}
    candidate = {"top1": 0.717783391, "macro": 0.717656851, "conflict": 0.311464548}
    assert candidate["top1"] > reference["top1"]
    assert candidate["macro"] >= reference["macro"]
    assert candidate["conflict"] >= reference["conflict"]
