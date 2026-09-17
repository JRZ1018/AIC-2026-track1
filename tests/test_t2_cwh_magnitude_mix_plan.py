"""Config-difference test for batch 6 item L (global loss-magnitude mixing).

Screening 2 (`scripts/analyze_classwise_weighting_capacity.py`) showed that the
within-class percentile that the soft weighting uses separates audited label conflicts
with AUC 0.642, while the *global* loss ranking reaches 0.790; at a 10% downweighting
budget the global ranking recovers 32.2% of true conflicts against 22.0%, and at 20% it
recovers 52.9% against 37.0%. The within-class percentile discards between-class loss
magnitude, which is where that signal lives.

L therefore tests one thing: blend the global loss ranking into the noise score at
``magnitude_mix=0.5``. It must be identical to the passing 12-epoch reference K in every
other respect, so that any Top-1 difference is attributable to that single variable.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiment"

REFERENCE = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_12ep_seed1_k.yaml"
CANDIDATE = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix50_12ep_seed1_l.yaml"
FULL_REFERENCE = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_12ep_full_seed1_k.yaml"

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


def test_reference_is_the_frozen_twelve_epoch_candidate() -> None:
    """Guard the reference so the comparison keeps its meaning."""
    assert _load(REFERENCE)["confidence_weighting"] == CWH_01
    assert _load(REFERENCE)["training"]["epochs"] == 12


def test_magnitude_mix_is_the_only_difference_from_the_reference() -> None:
    reference = _drop_identity(_load(REFERENCE))
    candidate = _drop_identity(_load(CANDIDATE))

    assert reference.pop("confidence_weighting") == CWH_01
    assert candidate.pop("confidence_weighting") == {**CWH_01, "magnitude_mix": 0.5}
    assert reference == candidate


def test_candidate_stays_a_single_seed_screening_run() -> None:
    """Screening must not silently become the full-retrain or multiseed stage."""
    experiment = _load(CANDIDATE)
    assert experiment["training_scope"] == "fixed_split_validation"
    assert experiment["training"]["seed"] == 20260910
    assert experiment["training"]["epochs"] == 12
    assert experiment["head_checkpoint"].endswith("20260911-p3-h1-linear-adam-e100-seed42/best.pt")


def test_candidate_output_dir_is_its_own_run_directory() -> None:
    experiment = _load(CANDIDATE)
    assert experiment["output_dir"].endswith(str(experiment["run_id"]))


def test_full_retrain_reference_has_no_magnitude_mix() -> None:
    """The historical full-train config must not gain the new variable by accident."""
    full = _load(FULL_REFERENCE)
    assert full["training_scope"] == "full_train_fixed_epochs"
    assert "magnitude_mix" not in full["confidence_weighting"]


def test_screening_evidence_file_exists_and_supports_the_choice() -> None:
    """The preregistration must cite a measurement, not an intuition.

    Both reports are produced by ``scripts/analyze_classwise_weighting_capacity.py`` and
    are local artifacts, so the test skips when they are absent rather than failing a
    clean checkout.
    """
    import json

    whole = ROOT / "artifacts" / "reports" / "screening2_signal_quality_wholesplit.json"
    validation = (
        ROOT / "artifacts" / "reports" / "screening2_classwise_weighting_capacity_validation.json"
    )
    if not whole.is_file() or not validation.is_file():
        return

    quality = json.loads(whole.read_text(encoding="utf-8"))
    assert quality["partition_filter"] == "all"
    assert quality["sample_count"] == 103218
    assert quality["conflict_sample_count"] == 2095

    # The whole point of item L: between-class loss magnitude carries the signal that the
    # within-class percentile throws away.
    assert quality["auc_global_loss"] > quality["auc_classwise_rank"] + 0.10
    assert quality["auc_global_loss"] > 0.75
    assert quality["auc_class_centered_loss"] < quality["auc_classwise_rank"] + 0.02

    by_budget = {row["budget"]: row for row in quality["budget_table"]}
    for budget in (0.05, 0.10, 0.20, 0.30, 0.50):
        row = by_budget[budget]
        assert row["global_loss_recall"] > row["classwise_rank_recall"] + 0.05
    assert by_budget[0.10]["global_loss_recall"] > 0.30
    assert by_budget[0.10]["classwise_rank_recall"] < 0.25

    # The within-class percentile forces an identical weight shape on every class, so its
    # richest bucket cannot concentrate conflicts the way a global ranking does.
    best_bucket = max(row["enrichment"] for row in quality["percentile_buckets"])
    assert best_bucket < 2.5

    report = json.loads(validation.read_text(encoding="utf-8"))
    signal = report["real_loss_signal"]
    assert signal["auc_loss_rank"] > report["auc_empirical"]
    assert signal["auc_loss_rank"] > 0.60
