"""CPU tests for the frozen C8 four-view promotion gate (docs/124)."""

from __future__ import annotations

import pytest

from scripts.run_stage2_c8_four_view_validation_queue import (
    FOLDS,
    MACRO_NO_LOSS_FLOOR_PP,
    MIN_PASSING_FOLDS,
    TAIL_NO_LOSS_FLOOR_PP,
    TOP1_GAIN_FLOOR_PP,
    evaluate_fold,
    evaluate_preregistered_gate,
    fold_outputs,
)


def _fold(
    tag: str,
    top1: float,
    *,
    macro: float = 0.50,
    tail: float = 0.50,
    ci95_low: float = 0.20,
    ci95_high: float | None = None,
) -> dict[str, object]:
    return evaluate_fold(
        tag=tag,
        top1_delta_pp=top1,
        macro_delta_pp=macro,
        tail_delta_pp=tail,
        ci95_low_pp=ci95_low,
        ci95_high_pp=top1 + 0.30 if ci95_high is None else ci95_high,
    )


def test_frozen_thresholds_are_the_preregistered_values() -> None:
    """Changing any of these constants is a new preregistration, not a retune."""

    assert TOP1_GAIN_FLOOR_PP == 0.30
    assert MACRO_NO_LOSS_FLOOR_PP == -0.25
    assert TAIL_NO_LOSS_FLOOR_PP == -0.50
    assert MIN_PASSING_FOLDS == 2


def test_every_fold_above_the_floor_promotes() -> None:
    gate = evaluate_preregistered_gate([_fold("fold1", 0.62), _fold("fold2", 0.44), _fold("fold3", 0.31)])
    assert gate["promote_candidate"] is True
    assert gate["decision"] == "promote_to_single_test_inference"
    assert gate["blocking_reasons"] == []


def test_two_passing_folds_and_one_small_but_positive_fold_promotes() -> None:
    gate = evaluate_preregistered_gate([_fold("fold1", 0.62), _fold("fold2", 0.44), _fold("fold3", 0.10)])
    assert gate["promote_candidate"] is True
    assert gate["passing_folds"] == ["fold1", "fold2"]
    assert gate["failed_folds"] == ["fold3"]


def test_a_negative_fold_blocks_promotion_even_with_two_passing_folds() -> None:
    gate = evaluate_preregistered_gate([_fold("fold1", 0.62), _fold("fold2", 0.44), _fold("fold3", -0.05)])
    assert gate["promote_candidate"] is False
    assert gate["decision"] == "stop_and_report"
    assert any("lost Top-1" in reason for reason in gate["blocking_reasons"])


def test_one_passing_fold_is_not_enough() -> None:
    gate = evaluate_preregistered_gate([_fold("fold1", 0.62), _fold("fold2", 0.20), _fold("fold3", 0.10)])
    assert gate["promote_candidate"] is False
    assert any("met the full per-fold gate" in reason for reason in gate["blocking_reasons"])


def test_macro_floor_blocks_promotion() -> None:
    gate = evaluate_preregistered_gate(
        [_fold("fold1", 0.62), _fold("fold2", 0.44), _fold("fold3", 0.35, macro=-0.30)]
    )
    assert gate["promote_candidate"] is False
    assert any("macro accuracy" in reason for reason in gate["blocking_reasons"])


def test_tail_floor_blocks_promotion() -> None:
    gate = evaluate_preregistered_gate(
        [_fold("fold1", 0.62), _fold("fold2", 0.44), _fold("fold3", 0.35, tail=-0.60)]
    )
    assert gate["promote_candidate"] is False
    assert any("tail class-mean recall" in reason for reason in gate["blocking_reasons"])


def test_a_non_positive_bootstrap_interval_cannot_count_as_passing() -> None:
    fold = _fold("fold1", 0.62, ci95_low=-0.01)
    assert fold["top1_gain_at_least_floor"] is True
    assert fold["bootstrap_interval_positive"] is False
    assert fold["fold_passed"] is False


def test_floors_are_inclusive_at_the_preregistered_boundary() -> None:
    fold = _fold("fold1", 0.30, macro=-0.25, tail=-0.50, ci95_low=0.0001)
    assert fold["fold_passed"] is True
    assert _fold("fold1", 0.2999, ci95_low=0.0001)["fold_passed"] is False
    # A zero-width or exactly-zero lower bound is not "positive".
    assert _fold("fold1", 0.30, ci95_low=0.0)["fold_passed"] is False


def test_gate_requires_at_least_one_fold() -> None:
    with pytest.raises(ValueError, match="at least one fold"):
        evaluate_preregistered_gate([])


def test_zero_recall_is_reported_as_non_veto_in_the_verdict() -> None:
    gate = evaluate_preregistered_gate([_fold("fold1", 0.62), _fold("fold2", 0.44), _fold("fold3", 0.31)])
    assert "not a standalone veto" in gate["zero_recall_policy"]
    assert gate["scope"] == "Stage 2 train-derived fixed validation only; no test access"


def test_the_historical_c7_gain_profile_would_now_promote_on_top1_but_the_gate_is_new() -> None:
    """Recorded C7 deltas (+0.69/+1.09/+0.94 pp Top-1, macro also positive in all folds).

    The old zero-recall veto that failed C7 was revoked for *future* experiments
    in docs/120/docs/121; docs/118's historical verdict is not rewritten, and this
    prior does not make any new candidate pass automatically.
    """

    gate = evaluate_preregistered_gate(
        [
            _fold("fold1", 0.6894, macro=0.5581, ci95_low=0.28),
            _fold("fold2", 1.0862, macro=1.0807, ci95_low=0.71),
            _fold("fold3", 0.9382, macro=0.9144, ci95_low=0.71),
        ]
    )
    assert gate["promote_candidate"] is True


def test_macro_is_floor_only_with_no_direction_requirement() -> None:
    """Effective gate is A's original one (docs/125 section 3).

    A stricter ">=2 folds with positive delta-macro" secondary confirmation was
    added by the user at 20:57 and withdrawn at 23:20, so a Top-1 gain whose
    macro merely stays above the floor must still promote.
    """

    gate = evaluate_preregistered_gate(
        [
            _fold("fold1", 0.62, macro=0.05),
            _fold("fold2", 0.44, macro=-0.02),
            _fold("fold3", 0.35, macro=-0.10),
        ]
    )
    assert gate["promote_candidate"] is True
    assert "macro_positive_folds" not in gate
    assert "min_macro_positive_folds" not in gate["thresholds"]
    assert gate["thresholds"]["macro_requirement"].startswith("floor only")
    assert "macro_requirement" in gate["thresholds"]


def test_macro_floor_still_blocks_a_head_driven_gain() -> None:
    """Without the direction rule the floor remains the macro guard."""

    gate = evaluate_preregistered_gate(
        [
            _fold("fold1", 0.62, macro=-0.30),
            _fold("fold2", 0.44, macro=0.10),
            _fold("fold3", 0.35, macro=0.10),
        ]
    )
    assert gate["promote_candidate"] is False
    assert any("macro accuracy" in reason for reason in gate["blocking_reasons"])


def test_frozen_fold_identities_and_output_paths_are_unique() -> None:
    tags = [fold["tag"] for fold in FOLDS]
    assert tags == ["fold1", "fold2", "fold3"]
    checkpoints = [fold["checkpoint_sha256"] for fold in FOLDS]
    assert len(set(checkpoints)) == len(checkpoints)
    diagnostics = [str(fold_outputs(tag)[0]) for tag in tags]
    reports = [str(fold_outputs(tag)[1]) for tag in tags]
    assert len(set(diagnostics)) == len(diagnostics)
    assert len(set(reports)) == len(reports)
    for directory in diagnostics:
        assert "artifacts\\diagnostics" in directory
    for report in reports:
        assert "artifacts\\reports" in report
    # The third fold overlaps the other two and must be disclosed, not hidden.
    assert "overlap" in FOLDS[2]["overlap_note"]
