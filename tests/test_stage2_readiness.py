from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pytest
import torch
import yaml

from aictrack1.data.class_mapping import ClassEntry, ClassMapping
from aictrack1.data.contracts import (
    load_dataset_config,
    validate_competition_stage_references,
)
from aictrack1.data.manifest import SampleRecord
from aictrack1.data.splits import build_grouped_stratified_split
from aictrack1.methods.confidence import ClasswiseHistoryWeighting


ROOT = Path(__file__).resolve().parents[1]
DATA_TEMPLATE = ROOT / "configs" / "data" / "stage2.template.yaml"
EXPERIMENT_TEMPLATE = (
    ROOT / "configs" / "experiment" / "p3_stage2_readiness.template.yaml"
)
STAGE1_FINAL_RUN_ID = (
    "20260919-ac-final-h1init-t2-last4-lr175e6-cwh-mw02-magmix75-16ep-full-seed1"
)


def _load_payload(path: Path, key: str) -> dict:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return document[key]


def test_stage2_templates_are_explicit_isolated_and_not_executable() -> None:
    dataset = _load_payload(DATA_TEMPLATE, "dataset")
    experiment = _load_payload(EXPERIMENT_TEMPLATE, "experiment")

    assert dataset["competition_stage"] == "stage2"
    assert dataset["contract_confirmed"] is False
    assert dataset["stage"] == "rematch"
    assert dataset["class_mapping_provenance"] == "official_file"
    assert dataset["submission"]["official_class_id_pattern"] is None
    assert experiment["competition_stage"] == "stage2"
    assert experiment["execution_enabled"] is False
    assert "DataSetStage2" in dataset["dataset_root"]
    assert "stage2" in experiment["dataset_config"]

    validate_competition_stage_references("stage2", dataset, context="dataset template")
    validate_competition_stage_references(
        "stage2", experiment, context="experiment template"
    )

    resolved = load_dataset_config(DATA_TEMPLATE)
    assert resolved.dataset_root == (ROOT / "DataSetStage2").resolve()
    assert resolved.class_mapping == (
        ROOT / "DataSetStage2" / "metadata" / "classes.csv"
    ).resolve()
    assert resolved.audit.output_root == (
        ROOT / "artifacts" / "audits" / "stage2"
    ).resolve()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_config", "../data/stage1.yaml"),
        ("manifest", "../../artifacts/audits/stage1/train/samples.csv"),
        ("feature_cache", "../../artifacts/features/stage1/train_openai_vit_b32"),
        ("head_checkpoint", "../../artifacts/experiments/stage1-h1/best.pt"),
        ("output_dir", "../../artifacts/experiments/p3_stage1_accidental"),
        ("dataset_root", "../../DataSetStage1"),
    ],
)
def test_stage2_guard_rejects_every_stage1_asset_class(field: str, value: str) -> None:
    with pytest.raises(ValueError, match="cross-stage reference rejected"):
        validate_competition_stage_references(
            "stage2", {field: value}, context="synthetic Stage 2 config"
        )


def test_stage2_guard_ignores_non_reference_history_text() -> None:
    validate_competition_stage_references(
        "stage2",
        {
            "notes": "Stage 1 evidence is historical context only",
            "manifest": "../../artifacts/audits/stage2/train/samples.csv",
        },
    )


def test_stage_guard_rejects_paths_in_reference_lists() -> None:
    with pytest.raises(ValueError, match=r"checkpoint\.1=.*stage1"):
        validate_competition_stage_references(
            "stage2",
            {
                "checkpoint": [
                    "../../artifacts/experiments/stage2-a/final.pt",
                    "../../artifacts/experiments/stage1-b/final.pt",
                ]
            },
        )


def test_stage_guard_uses_tokens_instead_of_substring_matches() -> None:
    validate_competition_stage_references(
        "stage2",
        {"output_dir": "../../artifacts/experiments/stage10-ablation"},
    )


def test_stage1_guard_rejects_stage2_reference() -> None:
    with pytest.raises(ValueError, match="references stage2"):
        validate_competition_stage_references(
            "stage1",
            {"manifest": "../../artifacts/audits/stage2/train/samples.csv"},
        )


def test_750_class_mapping_and_confidence_state_are_dynamic() -> None:
    mapping = ClassMapping(
        [ClassEntry(f"{class_id:04d}", class_id) for class_id in range(750)]
    )
    assert len(mapping) == 750
    assert mapping.to_internal("0749") == 749
    assert mapping.to_official(0) == "0000"

    labels = torch.arange(750).repeat_interleave(2)
    indices = torch.arange(labels.numel())
    state = ClasswiseHistoryWeighting(
        indices,
        labels,
        momentum=0.8,
        history_mix=0.5,
        min_weight=0.02,
        magnitude_mix=0.75,
    )
    diagnostics = state.update_epoch(
        indices,
        torch.linspace(0.1, 3.0, labels.numel()),
        torch.linspace(0.95, 0.05, labels.numel()),
        labels,
    )
    assert diagnostics["epochs_updated"] == 1
    assert torch.isfinite(state.weights).all()
    assert float(state.weights.min()) >= 0.02 - 1e-7


def test_rare_classes_keep_at_least_one_training_sample() -> None:
    records: list[SampleRecord] = []
    for class_id in range(750):
        sample_count = 2 if class_id < 100 else 5
        official = f"{class_id:04d}"
        for sample_index in range(sample_count):
            records.append(
                SampleRecord(
                    sample_id=f"{official}-{sample_index}",
                    relative_path=f"{official}/{sample_index}.jpg",
                    source_split="train",
                    observed_label=official,
                    internal_class_index=class_id,
                    official_class_id=official,
                    decode_status="ok",
                )
            )

    assignment = build_grouped_stratified_split(
        records,
        validation_fraction=0.2,
        seed=20260921,
        min_train_per_class=1,
    )
    train_counts = Counter(
        record.official_class_id
        for record in records
        if assignment[record.sample_id] == "train"
    )
    assert len(train_counts) == 750
    assert min(train_counts.values()) >= 1


def test_stage1_final_ledger_identity_is_frozen() -> None:
    ledger = ROOT / "submissions" / "ledger.csv"
    with ledger.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scored = [row for row in rows if row["online_score"]]
    top = max(scored, key=lambda row: float(row["online_score"]))

    assert top["online_score"] == "61.8016"
    assert top["run_id"] == STAGE1_FINAL_RUN_ID
    assert top["checkpoint_sha256"] == (
        "1979e4bbb1ace371dd2ed741ce604af2a267325f9bb0cef0739502d5df873ece"
    )
    assert top["config_hash"] == (
        "9d6cc1f43decb1020ce9e52eeb4e91afe0070f5143415c5e6a836a277243db0a"
    )
    assert top["manifest_path"] == (
        "submissions/20260919_ac_mw02_magmix75_16ep_full_final/"
        "submission_manifest.json"
    )
