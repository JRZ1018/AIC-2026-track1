from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_migration_smoke_is_isolated_and_bounded() -> None:
    text = (ROOT / "scripts/run_stage2_migration_smoke.py").read_text(encoding="utf-8")
    assert "renamed_input_dataset" in text
    assert '"--steps",\n        "1"' in text
    assert "competition_test_accessed" in text
    assert "migration_smoke_only" in text
    assert "TrainingLease.acquire" in text
    assert "--build-local-package" in text


def test_peft_smoke_can_build_validation_only_package() -> None:
    text = (ROOT / "scripts/smoke_peft.py").read_text(encoding="utf-8")
    assert "--build-local-package" in text
    assert "competition test data not accessed" in text
    assert "local validation inference" in text
