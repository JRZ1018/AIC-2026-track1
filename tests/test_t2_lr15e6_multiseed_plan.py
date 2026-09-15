from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "experiment"
FILES = [
    CONFIG_ROOT / "p3_stage1_h1init_t2_last4_lr15e6_6ep_seed2_b.yaml",
    CONFIG_ROOT / "p3_stage1_h1init_t2_last4_lr15e6_6ep_seed3_c.yaml",
]
FULL = CONFIG_ROOT / "p3_stage1_h1init_t2_last4_lr15e6_6ep_full_seed1_a_planned.yaml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["experiment"]


def test_confirmation_configs_are_fixed_split_same_candidate() -> None:
    seeds = []
    for path in FILES:
        exp = _load(path)
        assert exp["execution_enabled"] is True
        assert exp["training_scope"] == "fixed_split_validation"
        assert exp["peft"] == {"mode": "last_n_blocks", "lora_blocks": 4}
        assert exp["training"]["epochs"] == 6
        assert exp["training"]["visual_learning_rate"] == 0.0000015
        assert exp["training"]["head_learning_rate"] == 0.0
        assert exp["training"]["seed"] in {20260911, 20260912}
        seeds.append(exp["training"]["seed"])
    assert seeds == [20260911, 20260912]


def test_full_config_stays_disabled_until_confirmation_gate() -> None:
    exp = _load(FULL)
    assert exp["execution_enabled"] is False
    assert exp["training_scope"] == "full_train_fixed_epochs"
    assert exp["expected_train_count"] == 103218
    assert exp["training"]["epochs"] == 6
    assert exp["training"]["visual_learning_rate"] == 0.0000015
