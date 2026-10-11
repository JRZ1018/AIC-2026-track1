from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def pair():
    return [yaml.safe_load((ROOT / f"configs/experiment/stage2_p5_sce_fold1_3090_{tag}.yaml").read_text(encoding="utf-8"))["experiment"]
            for tag in ("control", "candidate")]


def test_p5_changes_only_loss_and_run_identity():
    control, candidate = pair()
    assert control["base_loss"] == {"kind": "gce", "q": 0.7}
    assert candidate["base_loss"] == {"kind": "sce", "alpha": 0.1, "beta": 1.0, "log_zero": -4.0}
    assert control["run_id"] != candidate["run_id"]
    for ex in (control, candidate):
        for key in ("base_loss", "run_id", "output_dir"):
            ex.pop(key)
    assert control == candidate


def test_p5_is_disabled_and_preserves_p4_control_recipe():
    historical = yaml.safe_load((ROOT / "configs/experiment/stage2_p4_gce_fold1_3090_q07_control.yaml").read_text(encoding="utf-8"))["experiment"]
    for ex in pair():
        assert ex["execution_enabled"] is False
        assert ex["competition_stage"] == "stage2"
        assert ex["training_scope"] == "fixed_split_validation"
        assert ex["training"]["epochs"] == 8
        assert ex["training"]["batch_size"] == 64
        assert ex["training"]["seed"] == 20260921
        assert ex["head_checkpoint"] == historical["head_checkpoint"]
        for key in ("training", "confidence_weighting", "peft", "runtime", "manifest", "split", "model_config"):
            assert ex[key] == historical[key]
