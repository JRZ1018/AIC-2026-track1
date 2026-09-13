from __future__ import annotations

from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "experiment"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))["experiment"]


@pytest.mark.parametrize("family", ["last3_r4", "last2_r8a16"])
@pytest.mark.parametrize("number,seed", [(1, 20260910), (2, 20260911), (3, 20260912)])
def test_b_promoted_capacity_configs_are_enabled_and_single_variable(
    family: str, number: int, seed: int
) -> None:
    baseline = _load(f"p3_stage1_h1init_peft_lora_last2_r4_3ep_seed{number}_a.yaml")
    candidate = _load(
        f"p3_stage1_h1init_peft_lora_{family}_3ep_seed{number}_b_planned.yaml"
    )

    assert candidate["execution_enabled"] is True
    assert candidate["training"] == baseline["training"]
    assert candidate["training"]["seed"] == seed
    assert candidate["head_checkpoint"] == baseline["head_checkpoint"]
    assert Path(candidate["output_dir"]).name == candidate["run_id"]
    assert "test" not in " ".join(str(value) for value in candidate.values()).lower()

    common = {key for key in baseline if baseline[key] == candidate.get(key)}
    assert set(baseline) - common == {"run_id", "output_dir", "peft"}
    if family == "last3_r4":
        assert candidate["peft"] == baseline["peft"] | {"lora_blocks": 3}
    else:
        assert candidate["peft"] == baseline["peft"] | {
            "lora_rank": 8,
            "lora_alpha": 16.0,
        }
        assert candidate["peft"]["lora_alpha"] / candidate["peft"]["lora_rank"] == 2.0
