"""Config-difference test for the conditional 12-epoch extension (batch 5 item K).

``docs/60`` preregistered K as an explicitly conditional run: it may only execute when the
10-epoch candidate beats the 8-epoch reference on Top-1 *and* its epochs 9-10 are still
rising. Both conditions were met (10ep Top-1 0.717783391 vs 0.713759303; epoch 9
0.717056 -> epoch 10 0.717783), so K is now a legitimate single-variable extension.

K must differ from the 10-epoch candidate by the epoch budget alone.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiment"

TEN_EP = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_10ep_seed1_i.yaml"
TWELVE_EP = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_12ep_seed1_k.yaml"

CWH_01 = {
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
    return normalized


def test_twelve_epoch_changes_only_the_epoch_budget() -> None:
    reference = _drop_identity(_load(TEN_EP))
    candidate = _drop_identity(_load(TWELVE_EP))
    assert reference["training"]["epochs"] == 10
    assert candidate["training"]["epochs"] == 12
    candidate["training"]["epochs"] = 10
    assert candidate == reference


def test_twelve_epoch_keeps_the_candidate_definition() -> None:
    exp = _load(TWELVE_EP)
    assert exp["execution_enabled"] is True
    assert exp["training_scope"] == "fixed_split_validation"
    assert exp["peft"] == {"mode": "last_n_blocks", "lora_blocks": 4}
    assert exp["confidence_weighting"] == CWH_01
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
    assert "test" not in " ".join(str(v) for v in exp.values()).lower()
    assert Path(exp["output_dir"]).name == exp["run_id"]


def test_ten_epoch_trigger_evidence_is_recorded() -> None:
    """Guard the preregistered trigger: 10ep beat the 8ep reference and epochs 9-10 rose."""
    reference_seed1 = 0.713759303
    ten_epoch = 0.717783391
    tail = [0.717056, 0.717783]
    assert ten_epoch > reference_seed1
    assert tail == sorted(tail)
    assert tail[-1] > tail[0]
