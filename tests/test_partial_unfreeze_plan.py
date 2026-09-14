from __future__ import annotations

from pathlib import Path

import yaml

from aictrack1.models.peft import configure_visual_peft

from tests.test_peft import _Visual

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "experiment"
COUNT1 = CONFIG_ROOT / "p3_stage1_h1init_last_block_count1_lr2e6_3ep_seed1_planned.yaml"
COUNT2 = CONFIG_ROOT / "p3_stage1_h1init_last_block_count2_lr2e6_3ep_seed1_planned.yaml"
FULL_H1 = "artifacts/experiments/20260912-b-final-h1-linear-full-e100-seed1/final.pt"


def _experiment(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["experiment"]


def test_partial_unfreeze_candidates_are_locked_and_single_variable() -> None:
    first = _experiment(COUNT1)
    second = _experiment(COUNT2)

    for candidate in (first, second):
        assert candidate["execution_enabled"] is False
        assert candidate["peft"]["mode"] == "last_n_blocks"
        assert candidate["training"]["head_learning_rate"] == 0.0
        assert candidate["training"]["epochs"] == 3
        assert candidate["training"]["visual_learning_rate"] == 2e-6
        assert FULL_H1 in candidate["head_checkpoint"]
        assert "test" not in " ".join(str(value) for value in candidate.values()).lower()
        assert Path(candidate["output_dir"]).name == candidate["run_id"]

    differences = {
        key
        for key in first
        if key not in {"run_id", "output_dir", "peft"} and first[key] != second[key]
    }
    assert differences == set()
    assert first["peft"]["lora_blocks"] == 1
    assert second["peft"]["lora_blocks"] == 2


def test_partial_unfreeze_block_counts_are_bounded_by_the_backbone() -> None:
    one = configure_visual_peft(_Visual(), "last_n_blocks", lora_blocks=1)
    two = configure_visual_peft(_Visual(), "last_n_blocks", lora_blocks=2)
    one_parameters = {
        name for name, parameter in _Visual().named_parameters() if name in one.trainable_parameter_names
    }
    two_parameters = {
        name for name, parameter in _Visual().named_parameters() if name in two.trainable_parameter_names
    }
    assert one_parameters < two_parameters
    assert all(
        name.startswith(("transformer.resblocks.0.", "transformer.resblocks.1."))
        for name in two_parameters
    )
    assert one.trainable_visual_parameters < two.trainable_visual_parameters


def test_candidate_recipe_matches_the_rank32_baseline_initializer_and_split() -> None:
    candidate = _experiment(COUNT1)
    baseline = _experiment(
        CONFIG_ROOT / "p3_stage1_h1init_peft_lora_last2_r32a64_3ep_full_seed1_a_planned.yaml"
    )
    for field in ("dataset_config", "model_config", "manifest", "split", "label_conflicts"):
        assert candidate[field] == baseline[field]
    assert candidate["head_checkpoint"] == baseline["head_checkpoint"]
    assert candidate["training"]["seed"] == baseline["training"]["seed"]
