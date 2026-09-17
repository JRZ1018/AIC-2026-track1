"""Config tests for batch 6 item O: the magnitude_mix sweep.

Item L showed the global-loss-mixing lever is worth +0.3204 online (60.7922, a new high),
but it was only ever measured at one point. The sweep must therefore change that single
coefficient and nothing else, so its readings stay comparable to the 0.5 reference.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiment"

REFERENCE = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix50_12ep_seed1_l.yaml"
SWEEP = {
    0.25: "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix25_12ep_seed1_q.yaml",
    0.5: REFERENCE,
    0.75: "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix75_12ep_seed1_o.yaml",
    1.0: "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix100_12ep_seed1_p.yaml",
}


def _load(name: str) -> dict:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["experiment"]


def _drop_identity(config: dict) -> dict:
    normalized = copy.deepcopy(config)
    normalized.pop("run_id")
    normalized.pop("output_dir")
    return normalized


def test_every_sweep_point_changes_only_the_coefficient() -> None:
    reference = _drop_identity(_load(REFERENCE))
    for mix, name in SWEEP.items():
        if mix == 0.5:
            continue
        candidate = _drop_identity(_load(name))
        assert candidate["confidence_weighting"]["magnitude_mix"] == mix
        candidate["confidence_weighting"]["magnitude_mix"] = 0.5
        candidate["confidence_weighting"] = reference["confidence_weighting"]
        assert candidate == reference, name


def test_all_sweep_points_share_the_frozen_reference_settings() -> None:
    for mix, name in SWEEP.items():
        experiment = _load(name)
        assert experiment["training_scope"] == "fixed_split_validation"
        assert experiment["training"]["epochs"] == 12
        assert experiment["training"]["seed"] == 20260910
        assert experiment["training"]["visual_learning_rate"] == 0.00000175
        assert experiment["peft"] == {"mode": "last_n_blocks", "lora_blocks": 4}
        assert experiment["head_checkpoint"].endswith(
            "20260911-p3-h1-linear-adam-e100-seed42/best.pt"
        )
        weighting = experiment["confidence_weighting"]
        assert weighting["min_weight"] == 0.1
        assert weighting["history_mix"] == 0.5
        assert weighting["magnitude_mix"] == mix
        assert experiment["output_dir"].endswith(str(experiment["run_id"]))


def test_the_coefficient_is_inside_the_supported_range() -> None:
    """The trainer rejects anything outside [0, 1]."""
    for mix in SWEEP:
        assert 0.0 <= mix <= 1.0
    assert max(SWEEP) == 1.0, "the sweep must include the pure-global endpoint"
    assert 0.5 in SWEEP, "the scored reference point must stay in the table"


def test_reference_local_reading_is_the_documented_one() -> None:
    """Anchor the baseline to the recorded run rather than to memory."""
    import json

    path = (
        ROOT
        / "artifacts"
        / "experiments"
        / "20260917-l-t2-last4-lr175e6-cwh-mw10-magmix50-12ep-seed1"
        / "metrics.json"
    )
    if not path.is_file():
        return
    metrics = json.loads(path.read_text(encoding="utf-8"))
    assert abs(metrics["best_top1"] - 0.721419573) < 1e-9
