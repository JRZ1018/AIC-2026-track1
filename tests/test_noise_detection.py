from __future__ import annotations

from aictrack1.evaluation.noise_detection import evaluate_noise_detection


def test_noise_detection_metrics() -> None:
    result = evaluate_noise_detection(
        [True, True, False, False],
        [True, False, True, False],
        ["0000", "0000", "0001", "0001"],
        [0.9, 0.2, 0.7, 0.1],
    )
    overall = result["overall"]
    assert overall["tp"] == 1
    assert overall["fp"] == 1
    assert overall["precision"] == 0.5
    assert overall["recall"] == 0.5
    assert overall["clean_retention"] == 0.5
    assert overall["auroc"] == 0.75


def test_noise_detection_rejects_length_mismatch() -> None:
    try:
        evaluate_noise_detection([True], [], ["0000"])
    except ValueError as exc:
        assert "equal length" in str(exc)
    else:
        raise AssertionError("expected ValueError")
