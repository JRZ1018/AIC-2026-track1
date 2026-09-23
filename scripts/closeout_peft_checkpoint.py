"""Create an auditable metrics closeout from an existing PEFT checkpoint.

This utility is for historical runs whose training checkpoint and exact config
survived but whose ordinary metrics finalization failed.  It never invents an
environment identity or wall-clock duration; missing original evidence remains
explicitly absent so the central index records ``unknown_legacy``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_closeout(
    *,
    checkpoint_path: Path,
    selected_checkpoint_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = raw.get("experiment", raw)
    if not isinstance(config, dict):
        raise ValueError("experiment config must be a mapping")
    run_id = str(config.get("run_id") or "")
    if not run_id or output_dir.name != run_id:
        raise ValueError("config run_id must match output directory")
    expected_config_hash = _sha256(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    selected = torch.load(selected_checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("config_sha256") != expected_config_hash:
        raise ValueError("checkpoint config hash does not match supplied config")
    if selected.get("config_sha256") != expected_config_hash:
        raise ValueError("selected checkpoint config hash does not match supplied config")
    history = checkpoint.get("history")
    epoch = int(checkpoint.get("epoch", 0))
    if not isinstance(history, list) or len(history) != epoch or epoch < 1:
        raise ValueError("checkpoint history is incomplete")
    best_top1 = float(checkpoint["best_top1"])
    observed_best = max(float(row["validation"]["all"]["top1"]) for row in history)
    if abs(best_top1 - observed_best) > 1e-12:
        raise ValueError("checkpoint best_top1 disagrees with history")
    provenance_fields = (
        "competition_stage",
        "class_mapping_sha256",
        "train_manifest_sha256",
        "split_sha256",
        "clip_weight_sha256",
        "num_classes",
    )
    provenance = {field: checkpoint[field] for field in provenance_fields}
    for field, value in provenance.items():
        if selected.get(field) != value:
            raise ValueError(f"selected checkpoint {field} mismatch")
    final_validation = history[-1]["validation"]
    first_confidence = history[0].get("confidence_weighting", {})
    next_quality = first_confidence.get("next_effective_quality", {})
    train_count = next_quality.get("observation_count")
    metrics: dict[str, Any] = {
        "run_id": run_id,
        "scope": (
            "historical Stage2 fixed-split PEFT closeout recovered from checkpoint; "
            "original environment finalization unavailable; test data not accessed"
        ),
        "training_scope": str(config.get("training_scope", "fixed_split_validation")),
        "selection_rule": str(selected.get("selection_rule", "best fixed-split validation Top-1")),
        "history": history,
        "best_top1": best_top1,
        "duration_seconds": None,
        "peak_vram_mib": None,
        "train_count": train_count,
        "validation_count": final_validation["all"]["sample_count"],
        "peft": config.get("peft", {}),
        "confidence_weighting": config.get("confidence_weighting", {}),
        "selected_checkpoint": selected_checkpoint_path.name,
        "selected_checkpoint_sha256": _sha256(selected_checkpoint_path),
        "best_checkpoint_sha256": _sha256(selected_checkpoint_path),
        "last_checkpoint_sha256": _sha256(checkpoint_path),
        "config_sha256": expected_config_hash,
        "split_sha256": provenance["split_sha256"],
        "manifest_sha256": provenance["train_manifest_sha256"],
        "checkpoint_provenance": provenance,
        "closeout": {
            "schema": "aictrack1.peft_checkpoint_closeout.v1",
            "source_checkpoint": checkpoint_path.name,
            "environment_status": "unknown_original_environment",
        },
    }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--selected-checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.resolve()
    selected_checkpoint_path = args.selected_checkpoint.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    metrics = build_closeout(
        checkpoint_path=checkpoint_path,
        selected_checkpoint_path=selected_checkpoint_path,
        config_path=config_path,
        output_dir=output_dir,
    )
    if args.write:
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            raise FileExistsError(f"refusing to overwrite existing metrics: {metrics_path}")
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        local_config = output_dir / "config.yaml"
        if not local_config.exists():
            shutil.copy2(config_path, local_config)
        (output_dir / "closeout_command.txt").write_text(
            " ".join(sys.argv) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "run_id": metrics["run_id"],
                "best_top1": metrics["best_top1"],
                "history_epochs": len(metrics["history"]),
                "environment_status": metrics["closeout"]["environment_status"],
                "mode": "written" if args.write else "check_only",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
