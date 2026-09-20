from __future__ import annotations

import pytest

from aictrack1.engine.capacity import GIB, estimate_stage2_capacity


def test_capacity_estimate_reports_conservative_shortfall() -> None:
    report = estimate_stage2_capacity(
        free_bytes=90 * GIB,
        stage1_dataset_bytes=20 * GIB,
        stage1_feature_bytes=0.12 * GIB,
        stage1_audit_bytes=0.05 * GIB,
    )
    assert report["scale"]["all_images"] == pytest.approx(186_139 / 128_185)
    assert report["minimum_archive_scenario"]["passes"] is True
    assert report["conservative_archive_scenario"]["passes"] is False
    assert report["decision"] == "CONDITIONAL_NO_GO"


def test_capacity_estimate_goes_green_with_sufficient_free_space() -> None:
    report = estimate_stage2_capacity(
        free_bytes=160 * GIB,
        stage1_dataset_bytes=20 * GIB,
        stage1_feature_bytes=0.12 * GIB,
        stage1_audit_bytes=0.05 * GIB,
    )
    assert report["conservative_archive_scenario"]["passes"] is True
    assert report["decision"] == "GO"


def test_capacity_estimate_rejects_invalid_interval() -> None:
    with pytest.raises(ValueError, match="interval"):
        estimate_stage2_capacity(
            free_bytes=100 * GIB,
            stage1_dataset_bytes=20 * GIB,
            stage1_feature_bytes=1,
            stage1_audit_bytes=1,
            archive_min_gib=30,
            archive_max_gib=10,
        )
