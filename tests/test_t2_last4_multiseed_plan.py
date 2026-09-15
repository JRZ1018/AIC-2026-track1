from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "experiment"
SEED1 = CONFIG_ROOT / "p3_stage1_overnight_t2_b4_lr1e6_8ep_seed1_b.yaml"
SEED2 = CONFIG_ROOT / "p3_stage1_h1init_t2_last4_lr1e6_8ep_seed2_a.yaml"
SEED3 = CONFIG_ROOT / "p3_stage1_h1init_t2_last4_lr1e6_8ep_seed3_c.yaml"
FULL = CONFIG_ROOT / "p3_stage1_h1init_t2_last4_lr1e6_8ep_full_seed1_a_planned.yaml"
FULL_ENABLED = CONFIG_ROOT / "p3_stage1_h1init_t2_last4_lr1e6_8ep_full_seed1_a.yaml"


def _experiment(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["experiment"]


def test_fixed_split_multiseed_configs_only_change_identity_and_seed() -> None:
    configs = [_experiment(path) for path in (SEED1, SEED2, SEED3)]
    ignored = {"run_id", "output_dir"}
    expected_seeds = [20260910, 20260911, 20260912]
    for candidate, expected_seed in zip(configs, expected_seeds, strict=True):
        assert candidate["execution_enabled"] is True
        assert candidate["training_scope"] == "fixed_split_validation"
        assert candidate["training"]["seed"] == expected_seed
        assert candidate["peft"] == {"mode": "last_n_blocks", "lora_blocks": 4}
        assert candidate["training"]["epochs"] == 8
        assert Path(candidate["output_dir"]).name == candidate["run_id"]

    reference = {key: value for key, value in configs[0].items() if key not in ignored}
    reference["training"] = {**reference["training"], "seed": None}
    for candidate in configs[1:]:
        comparable = {key: value for key, value in candidate.items() if key not in ignored}
        comparable["training"] = {**comparable["training"], "seed": None}
        assert comparable == reference


def test_full_config_is_disabled_until_multiseed_gate() -> None:
    candidate = _experiment(FULL)
    assert candidate["execution_enabled"] is False
    assert candidate["training_scope"] == "full_train_fixed_epochs"
    assert candidate["expected_train_count"] == 103218
    assert candidate["head_checkpoint"].endswith(
        "20260912-b-final-h1-linear-full-e100-seed1/final.pt"
    )
    assert candidate["peft"] == {"mode": "last_n_blocks", "lora_blocks": 4}
    assert candidate["training"]["epochs"] == 8
    assert "test" not in " ".join(str(value) for value in candidate.values()).lower()


def test_enabled_full_config_only_flips_execution_gate() -> None:
    planned = _experiment(FULL)
    enabled = _experiment(FULL_ENABLED)
    assert planned["execution_enabled"] is False
    assert enabled["execution_enabled"] is True
    assert {**enabled, "execution_enabled": False} == planned
