from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "experiment"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))["experiment"]


def test_final_head_uses_all_training_records_and_fixed_last_epoch() -> None:
    final = _load("p3_stage1_h1_linear_adam_e100_full_seed1.yaml")
    screened = _load("p3_stage1_h1_linear_adam_e100_seed42.yaml")

    assert final["training_scope"] == "full_train_fixed_epochs"
    assert final["expected_train_count"] == 103218
    assert final["feature_cache"] == screened["feature_cache"]
    assert final["num_classes"] == 500
    assert final["head"] == {"kind": "linear"}
    assert final["training"] == screened["training"] | {"seed": 20260910}
    assert Path(final["output_dir"]).name == final["run_id"]


def test_final_lora_is_the_accepted_recipe_with_fixed_two_epochs() -> None:
    final = _load("p3_stage1_h1init_peft_lora_only_2ep_full_seed1_b.yaml")
    screened = _load("p3_stage1_h1init_peft_lora_only_3ep_seed1_a_v2.yaml")

    assert final["execution_enabled"] is True
    assert final["training_scope"] == "full_train_fixed_epochs"
    assert final["expected_train_count"] == 103218
    assert final["peft"] == screened["peft"]
    expected_training = dict(screened["training"])
    expected_training["epochs"] = 2
    assert final["training"] == expected_training
    assert final["training"]["seed"] == 20260910
    assert final["training"]["head_learning_rate"] == 0.0
    assert final["head_checkpoint"].endswith(
        "20260912-b-final-h1-linear-full-e100-seed1/final.pt"
    )
    assert Path(final["output_dir"]).name == final["run_id"]
    assert "test" not in " ".join(str(value) for value in final.values()).lower()


def test_last_two_block_final_config_reuses_full_h1_and_fixed_three_epochs() -> None:
    baseline = _load("p3_stage1_h1init_peft_lora_only_2ep_full_seed1_b.yaml")
    candidate = _load("p3_stage1_h1init_peft_lora_last2_r4_3ep_full_seed1_a.yaml")

    assert candidate["training_scope"] == "full_train_fixed_epochs"
    assert candidate["expected_train_count"] == 103218
    assert candidate["head_checkpoint"] == baseline["head_checkpoint"]
    assert candidate["training"] == baseline["training"] | {"epochs": 3}
    assert candidate["peft"] == {
        "mode": "lora_last_n_attention",
        "lora_rank": 4,
        "lora_alpha": 8.0,
        "lora_blocks": 2,
    }
    assert Path(candidate["output_dir"]).name == candidate["run_id"]
    assert "test" not in " ".join(str(value) for value in candidate.values()).lower()


def test_rank8_final_config_only_promotes_validated_capacity() -> None:
    rank4 = _load("p3_stage1_h1init_peft_lora_last2_r4_3ep_full_seed1_a.yaml")
    rank8 = _load("p3_stage1_h1init_peft_lora_last2_r8a16_3ep_full_seed1_a.yaml")

    assert {key for key in rank4 if rank4[key] != rank8[key]} == {
        "run_id",
        "output_dir",
        "peft",
    }
    assert rank8["peft"] == rank4["peft"] | {
        "lora_rank": 8,
        "lora_alpha": 16.0,
    }
    assert rank8["peft"]["lora_alpha"] / rank8["peft"]["lora_rank"] == 2.0
    assert rank8["training_scope"] == "full_train_fixed_epochs"
    assert rank8["expected_train_count"] == 103218
    assert "test" not in " ".join(str(value) for value in rank8.values()).lower()


def test_final_wrapper_has_one_locked_public_command() -> None:
    script = (ROOT / "scripts" / "train_final_h1_lora.py").read_text(encoding="utf-8")
    assert "full_train_fixed_epochs" in script
    assert "103218" in script
    assert "20260910" in script
    assert "train_frozen_head.py" in script
    assert "train_peft.py" in script


def test_full_train_paths_emit_progress_and_checkpoint_each_epoch() -> None:
    head_script = (ROOT / "scripts" / "train_frozen_head.py").read_text(encoding="utf-8")
    peft_script = (ROOT / "scripts" / "train_peft.py").read_text(encoding="utf-8")

    head_loop = head_script[head_script.index('if training_scope == "full_train_fixed_epochs":', head_script.index("started =")) :]
    assert 'trainer.save_checkpoint(output / "last.pt")' in head_loop
    assert '"stage": "full_train_head"' in head_loop
    assert '"stage": "full_train_peft"' in peft_script
