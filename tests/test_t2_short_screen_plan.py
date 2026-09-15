from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "experiment"
FILES = sorted(CONFIG_ROOT.glob("p3_stage1_h1init_t2_last4_lr*e6_6ep_seed1_*.yaml"))


def test_short_screen_is_three_fixed_split_last4_learning_rates() -> None:
    assert len(FILES) == 3
    rates = set()
    run_ids = set()
    for path in FILES:
        exp = yaml.safe_load(path.read_text(encoding="utf-8"))["experiment"]
        assert exp["execution_enabled"] is True
        assert exp["training_scope"] == "fixed_split_validation"
        assert exp["peft"] == {"mode": "last_n_blocks", "lora_blocks": 4}
        assert exp["training"]["epochs"] == 6
        assert exp["training"]["seed"] == 20260910
        assert exp["training"]["head_learning_rate"] == 0.0
        assert "test" not in " ".join(str(value) for value in exp.values()).lower()
        rates.add(exp["training"]["visual_learning_rate"])
        run_ids.add(exp["run_id"])
    assert rates == {0.00000075, 0.00000125, 0.0000015}
    assert len(run_ids) == 3
