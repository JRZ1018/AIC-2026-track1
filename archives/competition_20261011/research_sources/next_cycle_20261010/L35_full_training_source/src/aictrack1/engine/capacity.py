"""Read-only capacity estimates for the Stage 2 data and training workspace."""

from __future__ import annotations

import math
from typing import Any


GIB = 1024**3


def estimate_stage2_capacity(
    *,
    free_bytes: int,
    stage1_dataset_bytes: int,
    stage1_feature_bytes: int,
    stage1_audit_bytes: int,
    archive_min_gib: float = 10.0,
    archive_max_gib: float = 30.0,
    checkpoint_budget_gib: float = 8.0,
    package_log_budget_gib: float = 2.0,
    low_water_gib: float = 20.0,
    stage1_train_count: int = 103_218,
    stage1_test_count: int = 24_967,
    stage2_train_count: int = 148_695,
    stage2_test_count: int = 37_444,
) -> dict[str, Any]:
    values = (
        free_bytes,
        stage1_dataset_bytes,
        stage1_feature_bytes,
        stage1_audit_bytes,
        stage1_train_count,
        stage1_test_count,
        stage2_train_count,
        stage2_test_count,
    )
    if any(value < 0 for value in values):
        raise ValueError("capacity inputs cannot be negative")
    if archive_min_gib < 0 or archive_max_gib < archive_min_gib:
        raise ValueError("archive size interval is invalid")
    if min(checkpoint_budget_gib, package_log_budget_gib, low_water_gib) < 0:
        raise ValueError("capacity budgets cannot be negative")
    stage1_total = stage1_train_count + stage1_test_count
    stage2_total = stage2_train_count + stage2_test_count
    if stage1_total == 0 or stage1_train_count == 0:
        raise ValueError("Stage 1 reference counts must be positive")
    data_scale = stage2_total / stage1_total
    train_scale = stage2_train_count / stage1_train_count
    extracted_gib = stage1_dataset_bytes / GIB * data_scale
    feature_gib = stage1_feature_bytes / GIB * train_scale
    audit_gib = stage1_audit_bytes / GIB * data_scale
    temporary_gib = max(10.0, extracted_gib * 0.5)

    def scenario(archive_gib: float) -> dict[str, Any]:
        peak = (
            archive_gib
            + extracted_gib
            + feature_gib
            + audit_gib
            + checkpoint_budget_gib
            + package_log_budget_gib
            + temporary_gib
        )
        required = peak + low_water_gib
        free_gib = free_bytes / GIB
        return {
            "archive_gib": archive_gib,
            "estimated_peak_new_gib": peak,
            "required_free_with_low_water_gib": required,
            "current_free_gib": free_gib,
            "margin_gib": free_gib - required,
            "passes": free_gib >= required,
        }

    minimum = scenario(archive_min_gib)
    conservative = scenario(archive_max_gib)
    return {
        "schema": "aictrack1.stage2_capacity.v1",
        "announced_counts_unverified": {
            "train": stage2_train_count,
            "test": stage2_test_count,
        },
        "scale": {"all_images": data_scale, "train_images": train_scale},
        "components_gib": {
            "estimated_extracted_dataset": extracted_gib,
            "estimated_feature_cache": feature_gib,
            "estimated_audit_outputs": audit_gib,
            "checkpoint_budget": checkpoint_budget_gib,
            "package_and_log_budget": package_log_budget_gib,
            "temporary_workspace": temporary_gib,
            "low_water_reserve": low_water_gib,
        },
        "minimum_archive_scenario": minimum,
        "conservative_archive_scenario": conservative,
        "recommended_free_gib": math.ceil(conservative["required_free_with_low_water_gib"]),
        "decision": "GO" if conservative["passes"] else "CONDITIONAL_NO_GO",
    }
