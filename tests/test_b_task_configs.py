from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiment"


def _load(name: str) -> dict:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["experiment"]


def _without_run_identity(config: dict) -> dict:
    normalized = copy.deepcopy(config)
    normalized.pop("run_id")
    normalized.pop("output_dir")
    normalized["training"].pop("seed")
    return normalized


def _without_method_identity(config: dict) -> dict:
    normalized = copy.deepcopy(config)
    normalized.pop("run_id")
    normalized.pop("output_dir")
    normalized.pop("confidence_weighting", None)
    return normalized


def test_b_seed_configs_only_change_run_identity_and_seed() -> None:
    pairs = [
        (
            "p3_stage1_peft_b3_lora_only_3ep_v2.yaml",
            "p3_stage1_peft_m0v2_seed2.yaml",
        ),
        (
            "p3_stage1_peft_b3_lora_only_3ep_v2.yaml",
            "p3_stage1_peft_m0v2_seed3.yaml",
        ),
        (
            "p3_stage1_m1_classwise_history_3ep.yaml",
            "p3_stage1_peft_m1_seed2.yaml",
        ),
        (
            "p3_stage1_m1_classwise_history_3ep.yaml",
            "p3_stage1_peft_m1_seed3.yaml",
        ),
    ]
    for base_name, variant_name in pairs:
        assert _without_run_identity(_load(base_name)) == _without_run_identity(
            _load(variant_name)
        )


def test_b_m0v2_and_m1_are_paired_within_each_seed() -> None:
    for suffix, expected_seed in (("seed2", 20260911), ("seed3", 20260912)):
        m0 = _load(f"p3_stage1_peft_m0v2_{suffix}.yaml")
        m1 = _load(f"p3_stage1_peft_m1_{suffix}.yaml")
        assert _without_method_identity(m0) == _without_method_identity(m1)
        assert m0["training"]["seed"] == expected_seed
        assert m1["training"]["seed"] == expected_seed
        assert "confidence_weighting" not in m0
        assert m1["confidence_weighting"] == {
            "enabled": True,
            "warmup_epochs": 1,
            "momentum": 0.8,
            "history_mix": 0.5,
            "min_weight": 0.2,
        }


def test_b_output_directories_and_run_ids_are_unique() -> None:
    names = [
        "p3_stage1_peft_m0v2_seed2.yaml",
        "p3_stage1_peft_m1_seed2.yaml",
        "p3_stage1_peft_m0v2_seed3.yaml",
        "p3_stage1_peft_m1_seed3.yaml",
    ]
    configs = [_load(name) for name in names]
    assert len({config["run_id"] for config in configs}) == len(configs)
    assert len({config["output_dir"] for config in configs}) == len(configs)
    for config in configs:
        assert Path(config["output_dir"]).name == config["run_id"]
