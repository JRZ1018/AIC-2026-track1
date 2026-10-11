from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    path = ROOT / "configs" / "experiment" / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))["experiment"]


def test_p8_residual_h1_configs_change_only_head_and_identity() -> None:
    pairs = [
        (
            "stage2_fold2_c5_h1_336_seed20260921.yaml",
            "stage2_p8_fold2_residual_h1_336_seed20260921.yaml",
        ),
        (
            "stage2_fold3_c5_h1_336_seed20260921.yaml",
            "stage2_p8_fold3_residual_h1_336_seed20260921.yaml",
        ),
    ]
    for control_name, candidate_name in pairs:
        control = _load(control_name)
        candidate = _load(candidate_name)
        assert candidate["execution_enabled"] is True
        assert candidate["head"] == {
            "kind": "residual_mlp",
            "hidden_dim": 256,
            "dropout": 0.1,
        }
        normalized_control = copy.deepcopy(control)
        normalized_candidate = copy.deepcopy(candidate)
        normalized_control["execution_enabled"] = True
        normalized_control["head"] = normalized_candidate["head"]
        for field in ("run_id", "output_dir"):
            normalized_control[field] = normalized_candidate[field]
        assert normalized_candidate == normalized_control
