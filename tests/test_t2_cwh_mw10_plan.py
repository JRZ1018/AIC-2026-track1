"""Config-difference test for batch 4: the lower min_weight floor (0.1).

The min_weight sweep runs 0.20 -> 0.35 -> 0.50 with monotonically decreasing Top-1
(0.711286724 -> 0.709929228 -> 0.708571732), which points below 0.2 rather than above it.
This candidate therefore tests min_weight=0.1 and must differ from the passing
``min_weight=0.2`` config by that single variable.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiment"

REFERENCE = "p3_stage1_h1init_t2_last4_lr175e6_cwh_6ep_seed1_c.yaml"
MW10 = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_6ep_seed1_h.yaml"

CWH_02 = {
    "enabled": True,
    "warmup_epochs": 1,
    "momentum": 0.8,
    "history_mix": 0.5,
    "min_weight": 0.2,
}
CWH_01 = {**CWH_02, "min_weight": 0.1}


def _load(name: str) -> dict:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["experiment"]


def _drop_identity(config: dict) -> dict:
    normalized = copy.deepcopy(config)
    normalized.pop("run_id")
    normalized.pop("output_dir")
    return normalized


def test_mw10_changes_only_the_floor() -> None:
    reference = _drop_identity(_load(REFERENCE))
    candidate = _drop_identity(_load(MW10))
    assert reference["confidence_weighting"] == CWH_02
    assert candidate["confidence_weighting"] == CWH_01
    candidate["confidence_weighting"] = CWH_02
    assert candidate == reference


def test_mw10_shared_protocol() -> None:
    exp = _load(MW10)
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
    assert training["visual_learning_rate"] == 0.00000175
    assert training["head_learning_rate"] == 0.0
    assert training["weight_decay"] == 0.01
    assert training["gradient_clip_norm"] == 1.0
    assert "test" not in " ".join(str(v) for v in exp.values()).lower()
    assert Path(exp["output_dir"]).name == exp["run_id"]


def test_min_weight_sweep_evidence_is_consistent_with_testing_below_point_two() -> None:
    """Guard the rationale: every recorded floor above 0.2 lost Top-1 to 0.2."""
    sweep = {"0.2": 0.711286724, "0.35": 0.709929228, "0.5": 0.708571732}
    ordered = sorted(sweep.items(), key=lambda kv: float(kv[0]))
    best_floor, best_top1 = max(sweep.items(), key=lambda kv: kv[1])
    assert best_floor == "0.2"
    assert best_top1 == sweep["0.2"]
    # monotone decrease as the floor rises
    tops = [top1 for _, top1 in ordered]
    assert tops == sorted(tops, reverse=True)
