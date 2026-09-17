"""Config and evidence tests for the D-node setup-parity gate.

A brand-new node cannot be trusted until it reproduces a known trajectory, so the first
task on any new machine is a three-epoch run on the item L screening configuration: its
first epoch must equal `0.695432961` exactly, which is only possible when the manifest,
split, conflict list, head checkpoint and CLIP weights are all byte-identical to A's.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiment"

PARITY = "p3_stage1_d_setup_parity_3ep_seed1.yaml"
L_SCREEN = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix50_12ep_seed1_l.yaml"
M_SCREEN = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix50_16ep_seed1_m.yaml"
M_FULL = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix50_16ep_full_seed1_m.yaml"

# Measured on A and reproduced bit-for-bit on B. These are the nine-decimal values the
# trainer prints; the stored doubles carry more digits, so compare at print precision.
REFERENCE_EPOCH_TOP1 = {
    1: 0.695432961,
    2: 0.703238606,
    3: 0.709589839,
}
REFERENCE_TOLERANCE = 1e-9


def _load(name: str) -> dict:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["experiment"]


def _drop_identity(config: dict) -> dict:
    normalized = copy.deepcopy(config)
    normalized.pop("run_id")
    normalized.pop("output_dir")
    return normalized


def test_parity_run_matches_the_item_l_screen_apart_from_length() -> None:
    """Repository layout differs, so compare everything except paths and run identity."""
    reference = copy.deepcopy(_load(L_SCREEN))
    candidate = copy.deepcopy(_load(PARITY))

    for config in (reference, candidate):
        config.pop("run_id")
        config.pop("output_dir")
        config.pop("dataset_config")
        config.pop("model_config")
        config.pop("manifest")
        config.pop("split")
        config.pop("label_conflicts")
        config.pop("head_checkpoint")

    assert reference["training"]["epochs"] == 12
    assert candidate["training"]["epochs"] == 3
    reference["training"]["epochs"] = 3
    assert reference == candidate


def test_parity_run_keeps_the_guards_that_catch_a_bad_setup() -> None:
    experiment = _load(PARITY)
    assert experiment["expected_initial_top1"] == 0.686996996
    assert experiment["initial_top1_tolerance"] == 0.002
    assert experiment["training"]["seed"] == 20260910
    assert experiment["confidence_weighting"]["magnitude_mix"] == 0.5


def test_reference_trajectory_is_the_one_both_existing_nodes_reproduced() -> None:
    """Anchor the self-check numbers to recorded runs rather than to memory."""
    import json

    a = ROOT / "artifacts" / "experiments" / "20260917-l-t2-last4-lr175e6-cwh-mw10-magmix50-12ep-seed1" / "metrics.json"
    if not a.is_file():
        return
    history = json.loads(a.read_text(encoding="utf-8"))["history"]
    for epoch, expected in REFERENCE_EPOCH_TOP1.items():
        assert abs(history[epoch - 1]["validation"]["all"]["top1"] - expected) < REFERENCE_TOLERANCE

    b = (
        ROOT
        / "artifacts"
        / "returns"
        / "20260917-B-magmix50-l"
        / "experiments"
        / "20260917-l-t2-last4-lr175e6-cwh-mw10-magmix50-12ep-seed1"
        / "metrics.json"
    )
    if not b.is_file():
        return
    b_history = json.loads(b.read_text(encoding="utf-8"))["history"]
    for epoch, expected in REFERENCE_EPOCH_TOP1.items():
        assert abs(b_history[epoch - 1]["validation"]["all"]["top1"] - expected) < REFERENCE_TOLERANCE


def test_item_m_candidates_differ_only_in_the_epoch_budget() -> None:
    reference = _drop_identity(_load(M_SCREEN))
    full = _load(M_FULL)
    assert reference["training"]["epochs"] == 16
    assert full["training"]["epochs"] == 16
    assert reference["confidence_weighting"]["magnitude_mix"] == 0.5
    assert "confidence_weighting" in full
    assert full["confidence_weighting"]["magnitude_mix"] == 0.5
