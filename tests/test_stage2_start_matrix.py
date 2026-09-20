from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from aictrack1.data.contracts import validate_competition_stage_references


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "experiment"
NAMES = (
    "stage2_h1_frozen_head.template.yaml",
    "stage2_peft_ce_last4.template.yaml",
    "stage2_peft_ac_last4.template.yaml",
)


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))["experiment"]


def test_stage2_start_templates_are_disabled_and_stage_isolated() -> None:
    for name in NAMES:
        config = _load(name)
        assert config["competition_stage"] == "stage2"
        assert config["execution_enabled"] is False
        assert config["run_id"].endswith("template-not-executable")
        assert "stage2" in config["dataset_config"]
        assert "stage1" not in " ".join(str(value).lower() for value in config.values())
        validate_competition_stage_references("stage2", config, context=name)


def test_h1_template_has_no_invented_class_count() -> None:
    h1 = _load(NAMES[0])
    assert h1["num_classes"] is None
    assert h1["head"] == {"kind": "linear"}
    assert h1["feature_cache"].endswith("stage2/train_openai_vit_b32")


def test_ce_and_ac_templates_have_exactly_one_algorithmic_change() -> None:
    ce = _load(NAMES[1])
    ac = _load(NAMES[2])
    normalized_ce = deepcopy(ce)
    normalized_ac = deepcopy(ac)
    for payload in (normalized_ce, normalized_ac):
        payload.pop("run_id")
        payload.pop("output_dir")
        payload.pop("confidence_weighting")
    assert normalized_ce == normalized_ac
    assert ce["confidence_weighting"] == {"enabled": False}
    assert ac["confidence_weighting"] == {
        "enabled": True,
        "warmup_epochs": 1,
        "momentum": 0.8,
        "history_mix": 0.5,
        "min_weight": 0.02,
        "magnitude_mix": 0.75,
    }


def test_peft_start_point_matches_frozen_stage1_reference_without_claiming_optimality() -> None:
    ce = _load(NAMES[1])
    assert ce["peft"] == {"mode": "last_n_blocks", "lora_blocks": 4}
    assert ce["training"] == {
        "epochs": 16,
        "batch_size": 64,
        "workers": 4,
        "seed": 20260921,
        "visual_learning_rate": 0.00000175,
        "head_learning_rate": 0.0,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
    }
