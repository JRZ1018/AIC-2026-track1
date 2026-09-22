from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from aictrack1.data.contracts import validate_competition_stage_references


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "experiment"
REFERENCE = "stage2_r4_ce_natural_8ep_reference.yaml"
BALANCED = "stage2_r4_ce_classbalanced_p05_8ep_seed20260921.yaml"
LOGIT = "stage2_r4_ce_logitadj_t05_8ep_seed20260921.yaml"
AC_REFERENCE = "stage2_r4_ac_8ep_reference.yaml"
AC_TAIL_FLOOR = "stage2_r4_ac_tailmeanfloor035_8ep_seed20260921.yaml"
AC_MIN_MASS = "stage2_r4_ac_minmass4_8ep_seed20260921.yaml"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))["experiment"]


def _identity_without_run(config: dict) -> dict:
    payload = deepcopy(config)
    payload.pop("run_id")
    payload.pop("output_dir")
    payload.pop("execution_enabled")
    return payload


def test_r4_reference_is_frozen_but_not_executable() -> None:
    reference = _load(REFERENCE)
    assert reference["execution_enabled"] is False
    assert reference["training"]["sampling"] == {"mode": "natural"}
    assert reference["training"]["epochs"] == 8
    validate_competition_stage_references("stage2", reference, context=REFERENCE)


def test_r4_balanced_sampling_is_one_variable_from_reference() -> None:
    reference = _identity_without_run(_load(REFERENCE))
    candidate = _identity_without_run(_load(BALANCED))
    sampling = candidate["training"].pop("sampling")
    reference["training"].pop("sampling")
    assert candidate == reference
    assert sampling == {"mode": "class_balanced", "class_balance_power": 0.5}


def test_r4_logit_adjustment_is_one_variable_from_reference() -> None:
    reference = _identity_without_run(_load(REFERENCE))
    candidate = _identity_without_run(_load(LOGIT))
    logit = candidate.pop("logit_adjustment")
    assert candidate == reference
    assert logit == {"enabled": True, "tau": 0.5}


def test_r4_candidates_are_enabled_and_stage_isolated() -> None:
    for name in (BALANCED, LOGIT):
        candidate = _load(name)
        assert candidate["execution_enabled"] is True
        assert candidate["competition_stage"] == "stage2"
        assert candidate["expected_initial_top1"] == 0.6447187066078186
        validate_competition_stage_references("stage2", candidate, context=name)


def test_r4_ac_protection_candidates_each_change_one_field_group() -> None:
    reference = _identity_without_run(_load(AC_REFERENCE))
    tail_floor = _identity_without_run(_load(AC_TAIL_FLOOR))
    min_mass = _identity_without_run(_load(AC_MIN_MASS))
    reference_confidence = reference.pop("confidence_weighting")
    tail_confidence = tail_floor.pop("confidence_weighting")
    mass_confidence = min_mass.pop("confidence_weighting")
    assert tail_floor == reference == min_mass
    assert tail_confidence == {
        **reference_confidence,
        "tail_class_fraction": 0.5,
        "tail_min_mean_weight": 0.35,
    }
    assert mass_confidence == {
        **reference_confidence,
        "min_effective_mass_per_class": 4.0,
    }


def test_r4_ac_protection_candidates_stay_locked_until_ce_screens_finish() -> None:
    for name in (AC_REFERENCE, AC_TAIL_FLOOR, AC_MIN_MASS):
        candidate = _load(name)
        assert candidate["execution_enabled"] is False
        assert candidate["competition_stage"] == "stage2"
        validate_competition_stage_references("stage2", candidate, context=name)
