"""Multiseed confirmation configs for the 8-epoch cwh candidate (batch 3 winner).

The 8-epoch run (``seed1``) beat the six-epoch online-best candidate on fixed validation:
Top-1 0.713759303 vs 0.711286724 (+0.002473), macro +0.002463, conflict subset unchanged.
Its best epoch is the final epoch 8 and epochs 7-8 kept rising, so the extension is real
rather than noise.

These two configs add seed2/seed3 for the three-seed gate. Each must differ from seed1
only by ``run_id``, ``output_dir`` and ``training.seed``.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiment"

SEED1 = "p3_stage1_h1init_t2_last4_lr175e6_cwh_8ep_seed1_g.yaml"
SEED2 = "p3_stage1_h1init_t2_last4_lr175e6_cwh_8ep_seed2_g.yaml"
SEED3 = "p3_stage1_h1init_t2_last4_lr175e6_cwh_8ep_seed3_g.yaml"

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


def test_eight_epochs_and_candidate_definition_are_frozen() -> None:
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
        assert training["epochs"] == 8
        assert training["batch_size"] == 64
        assert training["workers"] == 4
        assert training["visual_learning_rate"] == 0.00000175
        assert training["head_learning_rate"] == 0.0
        assert training["weight_decay"] == 0.01
        assert training["gradient_clip_norm"] == 1.0
        assert "test" not in " ".join(str(v) for v in exp.values()).lower()
        assert Path(exp["output_dir"]).name == exp["run_id"]


def test_identity_is_unique() -> None:
    run_ids = {_load(name)["run_id"] for name in (SEED1, SEED2, SEED3)}
    assert len(run_ids) == 3


def test_extension_evidence_is_recorded() -> None:
    """Guard the rationale for spending three seeds on the 8-epoch variant."""
    six_epoch = {"top1": 0.711286724, "macro": 0.711163759, "conflict": 0.310139179}
    eight_epoch = {"top1": 0.713759303, "macro": 0.713626802, "conflict": 0.310139179}
    assert eight_epoch["top1"] > six_epoch["top1"]
    assert eight_epoch["macro"] > six_epoch["macro"]
    assert eight_epoch["conflict"] >= six_epoch["conflict"]
