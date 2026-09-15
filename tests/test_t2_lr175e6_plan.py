from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "experiment"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))["experiment"]


def test_lr175e6_confirmation_is_frozen_and_test_free() -> None:
    seeds = []
    for suffix, seed in (("a", 20260910), ("b", 20260911), ("c", 20260912)):
        exp = _load(f"p3_stage1_h1init_t2_last4_lr175e6_6ep_seed{1 if suffix == 'a' else 2 if suffix == 'b' else 3}_{suffix}.yaml")
        assert exp["execution_enabled"] is True
        assert exp["training_scope"] == "fixed_split_validation"
        assert exp["peft"] == {"mode": "last_n_blocks", "lora_blocks": 4}
        assert exp["training"]["epochs"] == 6
        assert exp["training"]["visual_learning_rate"] == 0.00000175
        assert exp["training"]["seed"] == seed
        assert "test" not in " ".join(str(value) for value in exp.values()).lower()
        seeds.append(exp["training"]["seed"])
    assert seeds == [20260910, 20260911, 20260912]


def test_lr175e6_full_config_is_disabled() -> None:
    exp = _load("p3_stage1_h1init_t2_last4_lr175e6_6ep_full_seed1_a_planned.yaml")
    assert exp["execution_enabled"] is False
    assert exp["training_scope"] == "full_train_fixed_epochs"
    assert exp["expected_train_count"] == 103218
