from aictrack1.engine.readiness import evaluate_prelaunch


def _inputs() -> dict[str, object]:
    return {
        "repository_clean": True,
        "templates_disabled": True,
        "training_lock_free": True,
        "migration_smoke": {
            "migration_smoke_only": True,
            "competition_test_accessed": False,
            "checkpoint_restore_exact": True,
            "training_lock_released": True,
            "local_validation_package": {"record_count": 14},
        },
        "submission_simulation": {
            "row_count": 37_444,
            "unique_filenames": 37_444,
            "deterministic_csv": True,
            "deterministic_zip": True,
            "zip_member": "pred_results.csv",
        },
        "capacity": {
            "estimate": {
                "decision": "GO",
                "conservative_archive_scenario": {"margin_gib": 5.0},
            }
        },
        "account_confirmed": True,
    }


def test_prelaunch_go_requires_every_blocking_check() -> None:
    decision, checks = evaluate_prelaunch(**_inputs())
    assert decision == "GO"
    assert all(check.passed for check in checks)


def test_prelaunch_reports_capacity_and_account_blockers() -> None:
    inputs = _inputs()
    inputs["capacity"] = {
        "estimate": {
            "decision": "CONDITIONAL_NO_GO",
            "conservative_archive_scenario": {"margin_gib": -12.3},
        }
    }
    inputs["account_confirmed"] = False
    decision, checks = evaluate_prelaunch(**inputs)
    assert decision == "CONDITIONAL_NO_GO"
    failed = {check.name for check in checks if not check.passed}
    assert failed == {"capacity", "account_and_qualification"}


def test_prelaunch_rejects_incomplete_migration_or_submission_evidence() -> None:
    inputs = _inputs()
    inputs["migration_smoke"] = {"migration_smoke_only": True}
    inputs["submission_simulation"] = {"row_count": 37_444}
    decision, checks = evaluate_prelaunch(**inputs)
    assert decision == "CONDITIONAL_NO_GO"
    failed = {check.name for check in checks if not check.passed}
    assert {"migration_smoke", "submission_simulation"}.issubset(failed)
