"""Measure the current host and write a deterministic Stage 2 capacity estimate."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.capacity import estimate_stage2_capacity  # noqa: E402


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--archive-min-gib", type=float, default=10.0)
    parser.add_argument("--archive-max-gib", type=float, default=30.0)
    parser.add_argument("--checkpoint-budget-gib", type=float, default=8.0)
    parser.add_argument("--low-water-gib", type=float, default=20.0)
    args = parser.parse_args()
    usage = shutil.disk_usage(ROOT)
    measured = {
        "workspace_drive": ROOT.drive,
        "workspace_total_bytes": usage.total,
        "workspace_free_bytes": usage.free,
        "stage1_dataset_bytes": _tree_bytes(ROOT / "DataSetStage1"),
        "stage1_feature_bytes": _tree_bytes(ROOT / "artifacts" / "features" / "stage1"),
        "stage1_audit_bytes": _tree_bytes(ROOT / "artifacts" / "audits" / "stage1"),
        "current_experiment_bytes": _tree_bytes(ROOT / "artifacts" / "experiments"),
        "clip_weight_bytes": _tree_bytes(ROOT / "artifacts" / "weights" / "ViT-B-32.pt"),
    }
    estimate = estimate_stage2_capacity(
        free_bytes=usage.free,
        stage1_dataset_bytes=measured["stage1_dataset_bytes"],
        stage1_feature_bytes=measured["stage1_feature_bytes"],
        stage1_audit_bytes=measured["stage1_audit_bytes"],
        archive_min_gib=args.archive_min_gib,
        archive_max_gib=args.archive_max_gib,
        checkpoint_budget_gib=args.checkpoint_budget_gib,
        low_water_gib=args.low_water_gib,
    )
    payload = {"measured": measured, "estimate": estimate}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
