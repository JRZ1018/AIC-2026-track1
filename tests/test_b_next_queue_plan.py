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


@pytest.mark.parametrize(
    ("filename", "rank", "alpha", "blocks"),
    [
        ("p3_stage1_h1init_peft_lora_last2_r16a32_3ep_seed1_b_planned.yaml", 16, 32.0, 2),
        ("p3_stage1_h1init_peft_lora_last2_r32a64_3ep_seed1_b_planned.yaml", 32, 64.0, 2),
        ("p3_stage1_h1init_peft_lora_last2_r8a8_3ep_seed1_b_planned.yaml", 8, 8.0, 2),
        ("p3_stage1_h1init_peft_lora_last2_r8a32_3ep_seed1_b_planned.yaml", 8, 32.0, 2),
        ("p3_stage1_h1init_peft_lora_last3_r8a16_3ep_seed1_b_planned.yaml", 8, 16.0, 3),
        ("p3_stage1_h1init_peft_lora_last4_r8a16_3ep_seed1_b_planned.yaml", 8, 16.0, 4),
    ],
)
def test_next_lora_coarse_search_configs_are_disabled_and_bounded(
    filename: str, rank: int, alpha: float, blocks: int
) -> None:
    reference = _load(
        "p3_stage1_h1init_peft_lora_last2_r8a16_3ep_seed1_b_planned.yaml"
    )
    candidate = _load(filename)

    assert {key for key in reference if reference[key] != candidate[key]} == {
        "run_id",
        "execution_enabled",
        "output_dir",
        "peft",
    }
    assert candidate["execution_enabled"] is False
    assert candidate["peft"] == {
        "mode": "lora_last_n_attention",
        "lora_rank": rank,
        "lora_alpha": alpha,
        "lora_blocks": blocks,
    }
    assert candidate["training"] == reference["training"]
    assert candidate["head_checkpoint"] == reference["head_checkpoint"]
    assert candidate["manifest"] == reference["manifest"]
    assert candidate["split"] == reference["split"]
    assert "test" not in " ".join(str(value) for value in candidate.values()).lower()
