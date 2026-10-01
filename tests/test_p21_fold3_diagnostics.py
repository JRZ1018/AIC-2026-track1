"""Cheap rejection/boundary tests; no GPU, checkpoint, or competition data reads."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_stage2_p21_fold3_schedule_pair import gate_values
from run_stage2_p21_fold3_diagnostics import validate_history


def evidence(n=14):
    metrics = {"history": [{"epoch": i, "confidence_weighting": {
        "next_effective_quality": {"decision": "L1"}}} for i in range(1, n + 1)],
        "selected_checkpoint_sha256": "best", "last_checkpoint_sha256": "last"}
    check = {"complete_epochs": n, "best_sha256": "best", "last_sha256": "last"}
    return metrics, check


@pytest.mark.parametrize("n", [14, 20])
def test_complete_history(n):
    validate_history(n, *evidence(n))


@pytest.mark.parametrize("decision", ["L2", "unknown", None])
def test_bad_ac_rejected(decision):
    metrics, check = evidence()
    metrics["history"][3]["confidence_weighting"]["next_effective_quality"]["decision"] = decision
    with pytest.raises(ValueError, match="AC"):
        validate_history(14, metrics, check)


@pytest.mark.parametrize("change", ["missing", "duplicate", "reordered"])
def test_incomplete_epoch_history(change):
    metrics, check = evidence()
    if change == "missing":
        metrics["history"].pop(5)
    elif change == "duplicate":
        metrics["history"][5]["epoch"] = 5
    else:
        metrics["history"].reverse()
    with pytest.raises(ValueError, match="every complete epoch"):
        validate_history(14, metrics, check)


@pytest.mark.parametrize("key", ["complete_epochs", "best_sha256", "last_sha256"])
def test_receipt_mismatch(key):
    metrics, check = evidence()
    check[key] = "wrong"
    with pytest.raises(ValueError, match="receipt"):
        validate_history(14, metrics, check)


def test_exact_registered_boundaries():
    gate = gate_values({"delta": .001, "ci95_low": .00001}, 0., -.0025, 0.)
    assert all(gate.values())
    assert set(gate) == {"top1_at_least_plus_0_10pp", "group_ci95_lower_positive", "macro_nonnegative",
                         "tail_class_macro_at_least_minus_0_25pp", "ordinary_nonconflict_nonnegative"}


@pytest.mark.parametrize("index", range(5))
def test_each_failed_gate_rejects(index):
    values = [.001, .00001, 0., -.0025, 0.]
    values[index] = [.000999, 0., -.000001, -.002501, -.000001][index]
    gate = gate_values({"delta": values[0], "ci95_low": values[1]}, *values[2:])
    assert sum(not value for value in gate.values()) == 1
