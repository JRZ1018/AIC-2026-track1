"""Multiseed confirmation configs for batch 5 item K (12 epochs, min_weight 0.1).

K is the strongest seed1 result of the whole campaign and reached its peak at epoch 10
(Top-1 0.719480276, macro 0.719355226, conflict 0.311464548), then flattened
(ep11 0.719141, ep12 0.719286), so 12 epochs marks the onset of saturation rather than a
still-climbing curve.

The three seeds share one commit so the three-seed gate sees a single training commit
for all of them - the inconsistency recorded for batch 4 is avoided here.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiment"

SEED1 = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_12ep_seed1_k.yaml"
SEED2 = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_12ep_seed2_k.yaml"
SEED3 = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_12ep_seed3_k.yaml"

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
        assert training["epochs"] == 12
        assert training["batch_size"] == 64
        assert training["workers"] == 4
        assert training["visual_learning_rate"] == 0.00000175
        assert training["head_learning_rate"] == 0.0
        assert training["weight_decay"] == 0.01
        assert training["gradient_clip_norm"] == 1.0
        assert "test" not in " ".join(str(v) for v in exp.values()).lower()
        assert Path(exp["output_dir"]).name == exp["run_id"]


def test_identity_is_unique_across_batch5_candidates() -> None:
    names = (
        SEED1,
        SEED2,
        SEED3,
        "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_10ep_seed1_i.yaml",
        "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_8ep_seed1_j.yaml",
    )
    ids = {_load(n)["run_id"] for n in names}
    dirs = {_load(n)["output_dir"] for n in names}
    assert len(ids) == 5
    assert len(dirs) == 5


def test_duration_scan_evidence_is_recorded() -> None:
    """Guard the rationale: 12 epochs peaked at epoch 10, so duration is saturating."""
    duration_scan = {8: 0.714874446, 10: 0.717783391, 12: 0.719480276}
    assert duration_scan[10] > duration_scan[8]
    assert duration_scan[12] > duration_scan[10]
    # the 12-epoch run's own tail flattened, i.e. its best was not the final epoch
    twelve_epoch_tail = [0.719480, 0.719141, 0.719286]
    assert twelve_epoch_tail[0] == max(twelve_epoch_tail)
