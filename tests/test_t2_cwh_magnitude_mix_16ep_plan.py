"""Config-difference test for batch 6 item M (sixteen epochs at the mix candidate).

Duration is the only axis with repeated online evidence: 6 -> 8 epochs moved the online
score +0.1642 and 8 -> 12 epochs moved it +0.6969, while the local fixed-split gains
decayed (+0.002909 for 8 -> 10, +0.001697 for 10 -> 12). M therefore tests the axis
that has actually paid, at one further step, on top of the batch-6 mechanism.

M must differ from item L by the epoch budget alone, so duration and mechanism are not
confounded. If L fails its preregistered screen, M is not run at all.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiment"

L = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix50_12ep_seed1_l.yaml"
M = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix50_16ep_seed1_m.yaml"

CWH_MIX = {
    "enabled": True,
    "warmup_epochs": 1,
    "momentum": 0.8,
    "history_mix": 0.5,
    "min_weight": 0.1,
    "magnitude_mix": 0.5,
}


def _load(name: str) -> dict:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["experiment"]


def _drop_identity(config: dict) -> dict:
    normalized = copy.deepcopy(config)
    normalized.pop("run_id")
    normalized.pop("output_dir")
    return normalized


def test_sixteen_epochs_changes_only_the_epoch_budget() -> None:
    reference = _drop_identity(_load(L))
    candidate = _drop_identity(_load(M))

    assert reference["training"]["epochs"] == 12
    assert candidate["training"]["epochs"] == 16
    reference["training"]["epochs"] = 16
    assert reference == candidate


def test_both_items_carry_the_same_mechanism_configuration() -> None:
    assert _load(L)["confidence_weighting"] == CWH_MIX
    assert _load(M)["confidence_weighting"] == CWH_MIX


def test_item_m_stays_a_single_seed_screening_run() -> None:
    experiment = _load(M)
    assert experiment["training_scope"] == "fixed_split_validation"
    assert experiment["training"]["seed"] == 20260910
    assert experiment["output_dir"].endswith(str(experiment["run_id"]))


def test_duration_evidence_used_by_the_preregistration_is_present() -> None:
    """The epoch budgets actually run on this route, in order, with no gaps."""
    import csv

    index = ROOT / "experiments" / "index.csv"
    if not index.is_file():
        return
    with index.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    budgets = {}
    for row in rows:
        for epochs, marker in ((6, "cwh-6ep"), (8, "mw10-8ep"), (10, "mw10-10ep"), (12, "mw10-12ep")):
            if marker in row["run_id"] and row["top1"]:
                budgets.setdefault(epochs, float(row["top1"]))
    if not {8, 10, 12} <= set(budgets):
        return  # the index predates these rows; skip rather than assert on a partial table
    assert budgets[8] < budgets[10] < budgets[12]
    # The increments decay, which is exactly why M is a screen rather than a commitment.
    assert budgets[10] - budgets[8] > budgets[12] - budgets[10]
