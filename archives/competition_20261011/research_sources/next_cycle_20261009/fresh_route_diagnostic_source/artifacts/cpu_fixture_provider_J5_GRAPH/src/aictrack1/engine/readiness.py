"""Pure Stage 2 prelaunch go/no-go evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    passed: bool
    blocking: bool
    detail: str


def evaluate_prelaunch(
    *,
    repository_clean: bool,
    templates_disabled: bool,
    training_lock_free: bool,
    migration_smoke: Mapping[str, object],
    submission_simulation: Mapping[str, object],
    capacity: Mapping[str, object],
    account_confirmed: bool,
) -> tuple[str, tuple[ReadinessCheck, ...]]:
    package = migration_smoke.get("local_validation_package")
    smoke_ok = (
        migration_smoke.get("migration_smoke_only") is True
        and migration_smoke.get("competition_test_accessed") is False
        and migration_smoke.get("checkpoint_restore_exact") is True
        and migration_smoke.get("training_lock_released") is True
        and isinstance(package, Mapping)
        and int(package.get("record_count", 0)) > 0
    )
    submission_ok = (
        int(submission_simulation.get("row_count", 0)) == 37_444
        and int(submission_simulation.get("unique_filenames", 0)) == 37_444
        and submission_simulation.get("deterministic_csv") is True
        and submission_simulation.get("deterministic_zip") is True
        and submission_simulation.get("zip_member") == "pred_results.csv"
    )
    estimate = capacity.get("estimate")
    capacity_ok = isinstance(estimate, Mapping) and estimate.get("decision") == "GO"
    capacity_detail = "capacity report is missing"
    if isinstance(estimate, Mapping):
        conservative = estimate.get("conservative_archive_scenario", {})
        margin = conservative.get("margin_gib") if isinstance(conservative, Mapping) else None
        capacity_detail = (
            f"decision={estimate.get('decision')}, "
            f"conservative_margin_gib={margin}"
        )

    checks = (
        ReadinessCheck("repository_clean", repository_clean, True, "tracked worktree state"),
        ReadinessCheck("templates_disabled", templates_disabled, True, "all Stage 2 templates are inert"),
        ReadinessCheck("training_lock_free", training_lock_free, True, "repository training lease"),
        ReadinessCheck("migration_smoke", smoke_ok, True, "audit/split/train/restore/infer/package"),
        ReadinessCheck("submission_simulation", submission_ok, True, "37,444-row deterministic package"),
        ReadinessCheck("capacity", capacity_ok, True, capacity_detail),
        ReadinessCheck(
            "account_and_qualification",
            account_confirmed,
            True,
            "captain must confirm qualification and data/submission entries after login",
        ),
    )
    blocking_failures = [check for check in checks if check.blocking and not check.passed]
    return ("GO" if not blocking_failures else "CONDITIONAL_NO_GO"), checks
