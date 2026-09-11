from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiment"
SEEDS = {
    "seed42": 42,
    "seed2": 20260911,
    "seed3": 20260912,
}


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))["experiment"]


def test_fair_head_matrix_matches_teammate_optimizer_budget() -> None:
    run_ids = set()
    output_dirs = set()
    for suffix, seed in SEEDS.items():
        for method, head_kind in (("h1_linear", "linear"), ("h2_cosine", "cosine")):
            raw = _load(f"p3_stage1_{method}_adam_e100_{suffix}.yaml")
            training = raw["training"]
            assert raw["head"]["kind"] == head_kind
            assert training == {
                "epochs": 100,
                "batch_size": 4096,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "momentum": 0.9,
                "gradient_clip_norm": None,
                "seed": seed,
                "optimizer": "adam",
                "scheduler": "constant",
                "amp": False,
            }
            assert raw["device"] == "cuda"
            assert raw["num_classes"] == 500
            assert "train_openai_vit_b32" in raw["feature_cache"]
            assert Path(raw["output_dir"]).name == raw["run_id"]
            run_ids.add(raw["run_id"])
            output_dirs.add(raw["output_dir"])

    assert len(run_ids) == 6
    assert len(output_dirs) == 6


def test_linear_and_cosine_are_paired_within_seed() -> None:
    for suffix in SEEDS:
        linear = _load(f"p3_stage1_h1_linear_adam_e100_{suffix}.yaml")
        cosine = _load(f"p3_stage1_h2_cosine_adam_e100_{suffix}.yaml")
        assert linear["training"] == cosine["training"]
        for key in ("feature_cache", "device", "num_classes"):
            assert linear[key] == cosine[key]
