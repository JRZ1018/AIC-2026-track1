"""Config-difference test for the K seed1 same-commit re-run (audit only).

K's three-seed gate passed every core condition, but seed1 was recorded at commit
``5c609e8`` while seed2/seed3 ran at ``811a457`` (strict condition 8 wants one commit).
The config-hash check already proved the files identical (normalized SHA-256
``6382d5f292600b13eb7bc36800ded1cdae6c88a1bdc3e3e7ab5d8b5df604aa09`` matches seed1's
recorded ``config_sha256``); this v2 run closes the literal condition as well.

The v2 result is for audit and reproducibility only. It must never be used to cherry-pick
a more favourable number: whichever of seed1 / seed1-v2 is recorded, the three-seed mean
is computed from the same candidate definition.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiment"

SEED1 = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_12ep_seed1_k.yaml"
SEED1_V2 = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_12ep_seed1_v2_k.yaml"
SEED2 = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_12ep_seed2_k.yaml"
SEED3 = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_12ep_seed3_k.yaml"


def _load(name: str) -> dict:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["experiment"]


def _drop_identity(config: dict) -> dict:
    normalized = copy.deepcopy(config)
    normalized.pop("run_id")
    normalized.pop("output_dir")
    return normalized


def test_v2_reruns_the_same_candidate_under_a_new_identity() -> None:
    assert _drop_identity(_load(SEED1_V2)) == _drop_identity(_load(SEED1))


def test_v2_keeps_the_shared_seed() -> None:
    assert _load(SEED1_V2)["training"]["seed"] == 20260910
    assert _load(SEED1)["training"]["seed"] == 20260910


def test_v2_does_not_collide_with_any_other_k_run() -> None:
    names = (SEED1, SEED1_V2, SEED2, SEED3)
    ids = {_load(n)["run_id"] for n in names}
    dirs = {_load(n)["output_dir"] for n in names}
    assert len(ids) == 4
    assert len(dirs) == 4


def test_three_seed_gate_evidence_is_recorded() -> None:
    """K cleared every core condition; only the commit condition needed the v2 re-run."""
    rows = {
        20260910: (0.719480276107788, 0.719355225563049, 0.311464548110962),
        20260911: (0.718607604503632, 0.718502163887024, 0.314778000116348),
        20260912: (0.720692336559296, 0.720587968826294, 0.314115315675735),
    }
    baselines = {20260910: 0.705517292, 20260911: 0.705323398, 20260912: 0.705565810}
    top1 = [v[0] for v in rows.values()]
    macro = [v[1] for v in rows.values()]
    conflict = [v[2] for v in rows.values()]
    for seed, (t, _, _) in rows.items():
        assert t > baselines[seed], seed
    assert sum(top1) / 3 >= 0.714566386
    assert sum(macro) / 3 >= 0.713946720
    assert sum(conflict) / 3 >= 0.308801864
