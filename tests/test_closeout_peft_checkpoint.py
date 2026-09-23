from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "closeout_peft_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("closeout_peft_checkpoint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_closeout_preserves_missing_environment_and_validates_history(tmp_path: Path) -> None:
    run_id = "stage2-closeout"
    output = tmp_path / run_id
    output.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text(
        "experiment:\n"
        f"  run_id: {run_id}\n"
        "  competition_stage: stage2\n"
        "  training_scope: fixed_split_validation\n"
        "  peft:\n"
        "    mode: last_n_blocks\n",
        encoding="utf-8",
    )
    config_hash = MODULE._sha256(config)
    provenance = {
        "competition_stage": "stage2",
        "class_mapping_sha256": "1" * 64,
        "train_manifest_sha256": "2" * 64,
        "split_sha256": "3" * 64,
        "clip_weight_sha256": "4" * 64,
        "num_classes": 3,
    }
    history = [
        {
            "epoch": 1,
            "validation": {"all": {"top1": 0.5, "sample_count": 9}},
            "confidence_weighting": {
                "next_effective_quality": {"observation_count": 12}
            },
        }
    ]
    checkpoint = output / "last.pt"
    selected = output / "best.pt"
    torch.save(
        provenance
        | {
            "config_sha256": config_hash,
            "epoch": 1,
            "history": history,
            "best_top1": 0.5,
        },
        checkpoint,
    )
    torch.save(
        provenance
        | {
            "config_sha256": config_hash,
            "selection_rule": "best fixed-split validation Top-1",
        },
        selected,
    )

    metrics = MODULE.build_closeout(
        checkpoint_path=checkpoint,
        selected_checkpoint_path=selected,
        config_path=config,
        output_dir=output,
    )
    assert metrics["best_top1"] == 0.5
    assert metrics["train_count"] == 12
    assert metrics["validation_count"] == 9
    assert "environment" not in metrics
    assert metrics["closeout"]["environment_status"] == "unknown_original_environment"
