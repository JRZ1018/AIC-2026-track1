from __future__ import annotations

from pathlib import Path

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
