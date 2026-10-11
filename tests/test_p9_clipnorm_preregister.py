from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _experiment(relative: str) -> dict:
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))["experiment"]


def test_p9_changes_only_clip_norm_within_each_fold() -> None:
    pairs = (
        (
            "configs/experiment/stage2_p6_sce_fold2_a10_ac_on_control.yaml",
            "configs/experiment/stage2_p9_fold2_sce_ac_clipnorm10_candidate.yaml",
        ),
        (
            "configs/experiment/stage2_p5_sce_fold3_a10_candidate.yaml",
            "configs/experiment/stage2_p9_fold3_sce_ac_clipnorm10_candidate.yaml",
        ),
    )
    for control_path, candidate_path in pairs:
        control = _experiment(control_path)
        candidate = _experiment(candidate_path)
        ignored_identity = {"run_id", "output_dir"}
        changed = {
            key
            for key in control
            if key not in ignored_identity and control[key] != candidate[key]
        }
        assert changed == {"training"}
        changed_training = {
            key
            for key in control["training"]
            if control["training"][key] != candidate["training"][key]
        }
        assert changed_training == {"gradient_clip_norm"}
        assert control["training"]["gradient_clip_norm"] == 1.0
        assert candidate["training"]["gradient_clip_norm"] == 10.0
        assert candidate["execution_enabled"] is True


def test_p9_contract_is_fixed_for_both_folds() -> None:
    fold2 = _experiment("configs/experiment/stage2_p9_fold2_sce_ac_clipnorm10_candidate.yaml")
    fold3 = _experiment("configs/experiment/stage2_p9_fold3_sce_ac_clipnorm10_candidate.yaml")
    for config in (fold2, fold3):
        assert config["base_loss"] == {"kind": "sce", "alpha": 0.1, "beta": 1.0, "log_zero": -4.0}
        assert config["confidence_weighting"]["enabled"] is True
        assert config["training"]["epochs"] == 8
        assert config["training"]["batch_size"] == 64
        assert config["training"]["seed"] == 20260921
        assert config["training"]["gradient_clip_norm"] == 10.0
        assert config["visual_input_resolution"] == 336
        assert config["peft"] == {"mode": "last_n_blocks", "lora_blocks": 8}
