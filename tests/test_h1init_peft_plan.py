from __future__ import annotations

from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_h1_initialized_lora_config_is_safe_and_single_variable() -> None:
    planned_path = (
        ROOT
        / "configs"
        / "experiment"
        / "p3_stage1_h1init_peft_lora_only_3ep_seed1_planned.yaml"
    )
    m0_path = (
        ROOT / "configs" / "experiment" / "p3_stage1_peft_b3_lora_only_3ep_v2.yaml"
    )
    planned = yaml.safe_load(planned_path.read_text(encoding="utf-8"))["experiment"]
    m0 = yaml.safe_load(m0_path.read_text(encoding="utf-8"))["experiment"]

    assert planned["execution_enabled"] is False
    assert "h1-linear-adam-e100-seed42" in planned["head_checkpoint"]
    assert "a1_head_checkpoint" not in planned
    assert planned["peft"] == m0["peft"]
    assert planned["training"] == m0["training"]
    for field in ("dataset_config", "model_config", "manifest", "split", "label_conflicts"):
        assert planned[field] == m0[field]
    assert Path(planned["output_dir"]).name == planned["run_id"]


def test_enabled_h1_initialized_lora_config_matches_preregistered_recipe() -> None:
    config_root = ROOT / "configs" / "experiment"
    planned = yaml.safe_load(
        (config_root / "p3_stage1_h1init_peft_lora_only_3ep_seed1_planned.yaml").read_text(
            encoding="utf-8"
        )
    )["experiment"]
    enabled = yaml.safe_load(
        (config_root / "p3_stage1_h1init_peft_lora_only_3ep_seed1.yaml").read_text(
            encoding="utf-8"
        )
    )["experiment"]

    allowed_differences = {"execution_enabled", "run_id", "output_dir"}
    assert {key for key in planned if planned[key] != enabled[key]} == allowed_differences
    assert enabled["execution_enabled"] is True
    assert Path(enabled["output_dir"]).name == enabled["run_id"]


def test_a_run_only_changes_executor_identity_from_b_frozen_recipe() -> None:
    config_root = ROOT / "configs" / "experiment"
    b_run = yaml.safe_load(
        (config_root / "p3_stage1_h1init_peft_lora_only_3ep_seed1.yaml").read_text(
            encoding="utf-8"
        )
    )["experiment"]
    a_run = yaml.safe_load(
        (config_root / "p3_stage1_h1init_peft_lora_only_3ep_seed1_a.yaml").read_text(
            encoding="utf-8"
        )
    )["experiment"]

    assert {key for key in b_run if b_run[key] != a_run[key]} == {"run_id", "output_dir"}
    assert a_run["run_id"].startswith("20260911-a-")
    assert Path(a_run["output_dir"]).name == a_run["run_id"]


def test_a_v2_retry_only_changes_run_identity_after_normalization_fix() -> None:
    config_root = ROOT / "configs" / "experiment"
    first = yaml.safe_load(
        (config_root / "p3_stage1_h1init_peft_lora_only_3ep_seed1_a.yaml").read_text(
            encoding="utf-8"
        )
    )["experiment"]
    retry = yaml.safe_load(
        (config_root / "p3_stage1_h1init_peft_lora_only_3ep_seed1_a_v2.yaml").read_text(
            encoding="utf-8"
        )
    )["experiment"]

    assert {key for key in first if first[key] != retry[key]} == {"run_id", "output_dir"}
    assert retry["run_id"].endswith("-v2")
    assert Path(retry["output_dir"]).name == retry["run_id"]


def test_label_smoothing_candidate_is_disabled_and_single_variable() -> None:
    config_root = ROOT / "configs" / "experiment"
    baseline = yaml.safe_load(
        (config_root / "p3_stage1_h1init_peft_lora_only_3ep_seed1_a_v2.yaml").read_text(
            encoding="utf-8"
        )
    )["experiment"]
    candidate = yaml.safe_load(
        (
            config_root
            / "p3_stage1_h1init_peft_lora_label_smoothing_3ep_seed1.yaml"
        ).read_text(encoding="utf-8")
    )["experiment"]

    assert candidate["execution_enabled"] is False
    assert {key for key in baseline if baseline[key] != candidate[key]} == {
        "execution_enabled",
        "run_id",
        "output_dir",
        "training",
    }
    expected_training = dict(baseline["training"])
    expected_training["label_smoothing"] = 0.05
    assert candidate["training"] == expected_training
    assert Path(candidate["output_dir"]).name == candidate["run_id"]


def test_a_label_smoothing_run_only_unlocks_preregistered_candidate() -> None:
    config_root = ROOT / "configs" / "experiment"
    planned = yaml.safe_load(
        (
            config_root
            / "p3_stage1_h1init_peft_lora_label_smoothing_3ep_seed1.yaml"
        ).read_text(encoding="utf-8")
    )["experiment"]
    enabled = yaml.safe_load(
        (
            config_root
            / "p3_stage1_h1init_peft_lora_label_smoothing_3ep_seed1_a.yaml"
        ).read_text(encoding="utf-8")
    )["experiment"]

    assert {key for key in planned if planned[key] != enabled[key]} == {
        "execution_enabled"
    }
    assert enabled["execution_enabled"] is True


def test_a_label_smoothing_retry_only_changes_run_identity() -> None:
    config_root = ROOT / "configs" / "experiment"
    first = yaml.safe_load(
        (
            config_root
            / "p3_stage1_h1init_peft_lora_label_smoothing_3ep_seed1_a.yaml"
        ).read_text(encoding="utf-8")
    )["experiment"]
    retry = yaml.safe_load(
        (
            config_root
            / "p3_stage1_h1init_peft_lora_label_smoothing_3ep_seed1_a_v2.yaml"
        ).read_text(encoding="utf-8")
    )["experiment"]

    assert {key for key in first if first[key] != retry[key]} == {
        "run_id",
        "output_dir",
    }
    assert retry["run_id"].endswith("-v2")
    assert Path(retry["output_dir"]).name == retry["run_id"]


def test_last_two_attention_lora_candidate_only_expands_peft_scope() -> None:
    config_root = ROOT / "configs" / "experiment"
    baseline = yaml.safe_load(
        (config_root / "p3_stage1_h1init_peft_lora_only_3ep_seed1_a_v2.yaml").read_text(
            encoding="utf-8"
        )
    )["experiment"]
    candidate = yaml.safe_load(
        (
            config_root / "p3_stage1_h1init_peft_lora_last2_r4_3ep_seed1_a.yaml"
        ).read_text(encoding="utf-8")
    )["experiment"]

    assert {key for key in baseline if baseline[key] != candidate[key]} == {
        "run_id",
        "output_dir",
        "peft",
    }
    expected_peft = dict(baseline["peft"])
    expected_peft.update(mode="lora_last_n_attention", lora_blocks=2)
    assert candidate["peft"] == expected_peft
    assert candidate["execution_enabled"] is True


@pytest.mark.parametrize(
    ("filename", "seed", "suffix"),
    [
        ("p3_stage1_h1init_peft_lora_last2_r4_3ep_seed2_a.yaml", 20260911, "seed2"),
        ("p3_stage1_h1init_peft_lora_last2_r4_3ep_seed3_a.yaml", 20260912, "seed3"),
    ],
)
def test_last_two_attention_multiseed_configs_only_change_identity_and_seed(
    filename: str, seed: int, suffix: str
) -> None:
    config_root = ROOT / "configs" / "experiment"
    reference = yaml.safe_load(
        (
            config_root / "p3_stage1_h1init_peft_lora_last2_r4_3ep_seed1_a.yaml"
        ).read_text(encoding="utf-8")
    )["experiment"]
    candidate = yaml.safe_load((config_root / filename).read_text(encoding="utf-8"))[
        "experiment"
    ]

    assert {key for key in reference if reference[key] != candidate[key]} == {
        "run_id",
        "output_dir",
        "training",
    }
    expected_training = dict(reference["training"])
    expected_training["seed"] = seed
    assert candidate["training"] == expected_training
    assert candidate["run_id"].endswith(suffix)
    assert Path(candidate["output_dir"]).name == candidate["run_id"]


@pytest.mark.parametrize(
    ("filename", "seed", "suffix"),
    [
        ("p3_stage1_h1init_peft_lora_only_3ep_seed2_b.yaml", 20260911, "seed2"),
        ("p3_stage1_h1init_peft_lora_only_3ep_seed3_b.yaml", 20260912, "seed3"),
    ],
)
def test_b_multiseed_configs_only_change_identity_and_seed(
    filename: str, seed: int, suffix: str
) -> None:
    config_root = ROOT / "configs" / "experiment"
    reference = yaml.safe_load(
        (config_root / "p3_stage1_h1init_peft_lora_only_3ep_seed1_a_v2.yaml").read_text(
            encoding="utf-8"
        )
    )["experiment"]
    candidate = yaml.safe_load((config_root / filename).read_text(encoding="utf-8"))[
        "experiment"
    ]

    assert {key for key in reference if reference[key] != candidate[key]} == {
        "run_id",
        "output_dir",
        "training",
    }
    expected_training = dict(reference["training"])
    expected_training["seed"] = seed
    assert candidate["training"] == expected_training
    assert candidate["run_id"].endswith(suffix)
    assert Path(candidate["output_dir"]).name == candidate["run_id"]
