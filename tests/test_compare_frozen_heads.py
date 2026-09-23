from __future__ import annotations

from scripts.compare_frozen_heads import advancement_gate


def _report(*, top1: float, macro: float, tail: float, zero: int) -> dict:
    return {
        "validation": {
            "top1": top1,
            "macro_accuracy": macro,
            "zero_recall_class_count": zero,
            "strata": {"tail": {"accuracy": tail}},
        }
    }


def test_advancement_gate_accepts_balanced_gain_without_top1_regression() -> None:
    baseline = _report(top1=0.64, macro=0.63, tail=0.61, zero=7)
    candidate = _report(top1=0.6401, macro=0.632, tail=0.618, zero=3)

    result = advancement_gate(candidate, baseline)

    assert result["advance"] is True
    assert result["checks"] == {
        "top1_not_down_more_than_limit": True,
        "macro_improved": True,
        "tail_improved": True,
    }
    assert result["deltas"]["zero_recall_class_count"] == -4


def test_advancement_gate_rejects_large_top1_drop() -> None:
    baseline = _report(top1=0.64, macro=0.63, tail=0.61, zero=7)
    candidate = _report(top1=0.6379, macro=0.64, tail=0.62, zero=2)

    result = advancement_gate(candidate, baseline)

    assert result["advance"] is False
    assert result["checks"]["top1_not_down_more_than_limit"] is False
