"""Config-difference test for the item L full-train configuration.

The full run is what produces a submittable package. It must differ from the frozen
item K full config by the batch-6 mechanism alone, so that any online score difference
is attributable to `magnitude_mix` rather than to the head, the epoch budget or the
training scope.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiment"

FULL_K = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_12ep_full_seed1_k.yaml"
FULL_L = "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix50_12ep_full_seed1_l.yaml"


def _load(name: str) -> dict:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["experiment"]


def _drop_identity(config: dict) -> dict:
    normalized = copy.deepcopy(config)
    normalized.pop("run_id")
    normalized.pop("output_dir")
    return normalized


def test_full_run_changes_only_the_mechanism() -> None:
    reference = _drop_identity(_load(FULL_K))
    candidate = _drop_identity(_load(FULL_L))

    weight_k = reference["confidence_weighting"]
    weight_l = candidate["confidence_weighting"]
    assert weight_k["min_weight"] == weight_l["min_weight"] == 0.1
    assert weight_k["history_mix"] == weight_l["history_mix"] == 0.5
    assert "magnitude_mix" not in weight_k
    assert weight_l["magnitude_mix"] == 0.5

    reference["confidence_weighting"] = weight_l
    assert reference == candidate


def test_full_run_covers_the_whole_training_split() -> None:
    experiment = _load(FULL_L)
    assert experiment["training_scope"] == "full_train_fixed_epochs"
    assert experiment["expected_train_count"] == 103218
    assert experiment["training"]["epochs"] == 12


def test_full_run_reuses_the_same_frozen_head_as_the_scored_package() -> None:
    """The head is frozen, so it must not become a second variable."""
    experiment = _load(FULL_L)
    assert experiment["head_checkpoint"].endswith(
        "20260912-b-final-h1-linear-full-e100-seed1/final.pt"
    )
    assert experiment["training"]["head_learning_rate"] == 0.0
    assert experiment["training"]["visual_learning_rate"] == 0.00000175


def test_full_run_writes_its_own_run_directory() -> None:
    experiment = _load(FULL_L)
    assert experiment["output_dir"].endswith(str(experiment["run_id"]))
    assert "magmix50" in experiment["run_id"]
