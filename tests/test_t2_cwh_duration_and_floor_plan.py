"""Config-difference tests for the direction-screening round (2026-09-17).

Two open levers remain after the batch 1-4 results:

* **training duration** - the 8-epoch curve was still rising at its final epoch
  (0.711966 -> 0.712935 -> 0.713662 -> 0.713759 over epochs 5-8), so 10 epochs is the
  next step, not a blind extension.
* **the weighting floor** - min_weight 0.1 > 0.2 > 0.35 > 0.5 is monotone, so 0.1 may
  still not be the bottom.

Two candidates isolate these against the passing 8-epoch / min_weight=0.2 candidate
(``p3_stage1_h1init_t2_last4_lr175e6_cwh_8ep_seed1_g.yaml``):

* ``I`` = 10 epochs at min_weight 0.1  (duration + floor)
* ``J`` = 8 epochs  at min_weight 0.1  (floor only, at the known-good duration)

Each config must differ from the 8-epoch / 0.2 reference by exactly its preregistered
variables and nothing else.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiment"

REFERENCE = "p3_stage1_h1init_t2_last4_lr175e6_cwh_8ep_seed1_g.yaml"   # 8 epochs, mw 0.2
SIX_EP_REF = "p3_stage1_h1init_t2_last4_lr175e6_cwh_6ep_seed1_c.yaml"  # 6 epochs, mw 0.2
TEN_EP = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_10ep_seed1_i.yaml"
EIGHT_EP = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_8ep_seed1_j.yaml"

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


def test_reference_is_eight_epochs_at_floor_point_two() -> None:
    exp = _load(REFERENCE)
    assert exp["training"]["epochs"] == 8
    assert exp["confidence_weighting"] == CWH_02
    assert exp["training"]["visual_learning_rate"] == 0.00000175


def test_ten_epoch_candidate_changes_only_floor_and_duration() -> None:
    reference = _drop_identity(_load(REFERENCE))
    candidate = _drop_identity(_load(TEN_EP))
    assert candidate["confidence_weighting"] == CWH_01
    assert candidate["training"]["epochs"] == 10
    # normalise both preregistered variables back and everything else must match
    candidate["confidence_weighting"] = CWH_02
    candidate["training"]["epochs"] = 8
    assert candidate == reference


def test_eight_epoch_floor_candidate_changes_only_the_floor() -> None:
    """J isolates the floor at the known-good 8-epoch duration."""
    reference = _drop_identity(_load(REFERENCE))
    candidate = _drop_identity(_load(EIGHT_EP))
    assert candidate["training"]["epochs"] == 8
    assert candidate["confidence_weighting"] == CWH_01
    candidate["confidence_weighting"] = CWH_02
    assert candidate == reference


def test_floor_scan_evidence_justifies_testing_below_point_two() -> None:
    """Guard the rationale: every recorded floor above 0.2 lost Top-1 to 0.2 (6 epochs)."""
    sweep = {"0.10": 0.712159395, "0.20": 0.711286724, "0.35": 0.709929228, "0.50": 0.708571732}
    above = {k: v for k, v in sweep.items() if float(k) >= 0.2}
    ordered = sorted(above.items(), key=lambda kv: float(kv[0]))
    tops = [top1 for _, top1 in ordered]
    assert tops == sorted(tops, reverse=True), "recorded floors above 0.2 must degrade monotonically"
    assert sweep["0.10"] > sweep["0.20"], "0.1 currently beats 0.2, so the floor is not yet bottomed"


def test_duration_scan_evidence_justifies_testing_ten_epochs() -> None:
    """Guard the rationale: the 8-epoch curve was still rising at its last epoch."""
    eight_epoch_tail = [0.711966, 0.712935, 0.713662, 0.713759]
    assert eight_epoch_tail == sorted(eight_epoch_tail), "8-epoch tail must be monotonically rising"
    assert eight_epoch_tail[-1] > eight_epoch_tail[0]


def test_shared_protocol_is_frozen() -> None:
    for name in (TEN_EP, EIGHT_EP):
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
        assert "test" not in " ".join(str(v) for v in exp.values()).lower()
        assert Path(exp["output_dir"]).name == exp["run_id"]


def test_identity_is_unique_and_does_not_collide_with_prior_runs() -> None:
    ids = {_load(name)["run_id"] for name in (TEN_EP, EIGHT_EP, REFERENCE, SIX_EP_REF)}
    assert len(ids) == 4
    dirs = {_load(name)["output_dir"] for name in (TEN_EP, EIGHT_EP, REFERENCE, SIX_EP_REF)}
    assert len(dirs) == 4
