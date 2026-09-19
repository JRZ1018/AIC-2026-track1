"""Config tests for the 2026-09-18 follow-up queue (items P, R, S, T).

The fixed split was shown to have a resolution floor: mix 0.75 scored exactly equal to mix
0.50 locally (0.721419573 both, delta 0.000000000000) yet gained +0.2083 online. So these
runs go straight to full train one variable at a time, and each must be provably identical
to the then-current best (61.0005) apart from its single change.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiment"

BEST = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix75_12ep_full_seed1_o.yaml"
P = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix100_12ep_full_seed1_p.yaml"
R = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix75_12ep_full_seed2_r.yaml"
S = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw05_magmix75_12ep_full_seed1_s.yaml"
T = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix75_16ep_full_seed1_t.yaml"

ALL = (BEST, P, R, S, T)


def _load(name: str) -> dict:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["experiment"]


def _drop_identity(config: dict) -> dict:
    normalized = copy.deepcopy(config)
    normalized.pop("run_id")
    normalized.pop("output_dir")
    return normalized


def test_queue_sits_on_the_full_train_scope_with_the_scored_head() -> None:
    for name in ALL:
        experiment = _load(name)
        assert experiment["training_scope"] == "full_train_fixed_epochs", name
        assert experiment["expected_train_count"] == 103218, name
        assert experiment["head_checkpoint"].endswith(
            "20260912-b-final-h1-linear-full-e100-seed1/final.pt"
        ), name
        assert experiment["training"]["head_learning_rate"] == 0.0, name
        assert experiment["training"]["visual_learning_rate"] == 0.00000175, name
        assert experiment["peft"] == {"mode": "last_n_blocks", "lora_blocks": 4}, name


def test_p_extends_only_the_coefficient() -> None:
    reference = _drop_identity(_load(BEST))
    candidate = _drop_identity(_load(P))
    assert reference["confidence_weighting"]["magnitude_mix"] == 0.75
    assert candidate["confidence_weighting"]["magnitude_mix"] == 1.0
    candidate["confidence_weighting"]["magnitude_mix"] = 0.75
    assert candidate == reference


def test_r_is_the_noise_scale_probe_and_changes_only_the_seed() -> None:
    """Without this, a +0.2 difference cannot be separated from run-to-run noise."""
    reference = _drop_identity(_load(BEST))
    candidate = _drop_identity(_load(R))
    assert reference["training"]["seed"] == 20260910
    assert candidate["training"]["seed"] == 20260912
    candidate["training"]["seed"] = 20260910
    assert candidate == reference


def test_s_changes_only_the_weighting_floor() -> None:
    reference = _drop_identity(_load(BEST))
    candidate = _drop_identity(_load(S))
    assert reference["confidence_weighting"]["min_weight"] == 0.1
    assert candidate["confidence_weighting"]["min_weight"] == 0.05
    candidate["confidence_weighting"]["min_weight"] = 0.1
    assert candidate == reference


def test_t_changes_only_the_epoch_budget() -> None:
    reference = _drop_identity(_load(BEST))
    candidate = _drop_identity(_load(T))
    assert reference["training"]["epochs"] == 12
    assert candidate["training"]["epochs"] == 16
    candidate["training"]["epochs"] = 12
    assert candidate == reference


def test_every_queue_member_writes_its_own_run_directory() -> None:
    seen = set()
    for name in ALL:
        experiment = _load(name)
        assert experiment["output_dir"].endswith(str(experiment["run_id"])), name
        assert experiment["run_id"] not in seen, name
        seen.add(experiment["run_id"])


def test_historical_queue_baseline_is_the_one_that_scored_61_0005() -> None:
    """Anchor the historical queue baseline without asserting it remains global best."""
    import csv

    ledger = ROOT / "submissions" / "ledger.csv"
    if not ledger.is_file():
        return
    with ledger.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row["run_id"] == _load(BEST)["run_id"]]
    assert len(matches) == 1
    assert matches[0]["online_score"] == "61.0005"
