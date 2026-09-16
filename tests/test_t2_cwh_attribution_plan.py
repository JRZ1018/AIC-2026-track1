"""Config-difference tests for batch 2: mechanism attribution and min_weight search.

Batch 2 answers two questions at once:

* direction 2 (attribution): is the large gain driven by the 1.75e-6 learning rate, or by the
  classwise-history weighting?  ``lr15e6_cwh`` holds the weighting fixed and drops the learning
  rate back to the 1.5e-6 baseline value.
* direction 1 (parameter search): does a weaker floor ``min_weight=0.35`` beat 0.2 at 1.75e-6?

Both candidates must differ from their reference config by exactly the one preregistered
variable.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiment"

# reference configs already in the repo
LR15_BASE = "p3_stage1_h1init_t2_last4_lr15e6_6ep_seed1_c.yaml"
LR175_CWH = "p3_stage1_h1init_t2_last4_lr175e6_cwh_6ep_seed1_c.yaml"

# batch 2 candidates
LR15_CWH = "p3_stage1_h1init_t2_last4_lr15e6_cwh_6ep_seed1_d.yaml"
LR175_CWH_MW35 = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw35_6ep_seed1_e.yaml"

CWH_02 = {
    "enabled": True,
    "warmup_epochs": 1,
    "momentum": 0.8,
    "history_mix": 0.5,
    "min_weight": 0.2,
}
CWH_035 = {**CWH_02, "min_weight": 0.35}


def _load(name: str) -> dict:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["experiment"]


def _drop_identity(config: dict) -> dict:
    normalized = copy.deepcopy(config)
    normalized.pop("run_id")
    normalized.pop("output_dir")
    return normalized


def test_attribution_candidate_isolates_the_weighting_at_the_baseline_learning_rate() -> None:
    """lr15e6+cwh must equal the lr1.5e-6 CE reference plus the weighting block, nothing else."""
    reference = _drop_identity(_load(LR15_BASE))
    candidate = _drop_identity(_load(LR15_CWH))
    assert "confidence_weighting" not in reference
    assert candidate.pop("confidence_weighting") == CWH_02
    assert candidate == reference


def test_attribution_candidate_keeps_the_baseline_learning_rate() -> None:
    assert _load(LR15_CWH)["training"]["visual_learning_rate"] == 0.0000015
    assert _load(LR15_BASE)["training"]["visual_learning_rate"] == 0.0000015


def test_min_weight_candidate_changes_only_the_floor() -> None:
    """mw35 must equal the passing 1.75e-6+cwh config apart from run identity and min_weight."""
    reference = _drop_identity(_load(LR175_CWH))
    candidate = _drop_identity(_load(LR175_CWH_MW35))
    assert reference["confidence_weighting"] == CWH_02
    assert candidate["confidence_weighting"] == CWH_035
    # Normalise both sides to the same floor, then everything else must match exactly.
    candidate["confidence_weighting"] = CWH_02
    assert candidate == reference


def test_batch2_shared_protocol() -> None:
    for name in (LR15_CWH, LR175_CWH_MW35):
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
        assert training["epochs"] == 6
        assert training["batch_size"] == 64
        assert training["workers"] == 4
        assert training["seed"] == 20260910
        assert training["head_learning_rate"] == 0.0
        assert training["weight_decay"] == 0.01
        assert training["gradient_clip_norm"] == 1.0


def test_batch2_identity_is_unique_and_test_free() -> None:
    run_ids = set()
    for name in (LR15_CWH, LR175_CWH_MW35):
        exp = _load(name)
        assert "test" not in " ".join(str(value) for value in exp.values()).lower()
        assert Path(exp["output_dir"]).name == exp["run_id"]
        run_ids.add(exp["run_id"])
    assert len(run_ids) == 2
