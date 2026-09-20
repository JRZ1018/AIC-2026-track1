"""Run the isolated B05 migration smoke without touching competition test data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.audit import write_audit_report
from aictrack1.data.contracts import load_dataset_config
from aictrack1.data.manifest import read_manifest
from aictrack1.data.splits import build_grouped_stratified_split, write_split
from aictrack1.data.synthetic import generate_synthetic_dataset, synthetic_config_from_mapping
from aictrack1.engine.training_queue import (
    DEFAULT_LOCK_RELATIVE_PATH,
    TrainingLease,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_smoke(command: list[str], output_dir: Path, log_path: Path) -> int:
    lease = TrainingLease.acquire(
        ROOT / DEFAULT_LOCK_RELATIVE_PATH,
        owner="b05-stage2-migration-smoke",
        config=str(command[command.index("--dataset-config") + 1]),
        output_dir=str(output_dir),
        command=command,
    )
    process: subprocess.Popen[str] | None = None
    completed = False
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            lease.attach_trainer(process.pid)
            return_code = process.wait()
            completed = True
    finally:
        if completed or process is None:
            lease.release()
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    if run_root.exists():
        raise FileExistsError(f"migration smoke run root already exists: {run_root}")
    if not run_root.is_relative_to(ROOT):
        raise ValueError("migration smoke run root must be inside the repository")
    run_root.mkdir(parents=True)
    dataset_root = run_root / "renamed_input_dataset"
    audit_root = run_root / "audit"
    flow_root = run_root / "flow"
    config_path = run_root / "migration_smoke_dataset.yaml"
    split_path = flow_root / "split.csv"
    output_dir = flow_root / "peft_one_step"
    log_path = run_root / "peft_smoke.log"

    document = {
        "synthetic": {
            "output_root": str(dataset_root),
            "seed": 20260920,
            "num_classes": 7,
            "classes_per_family": 3,
            "max_samples_per_class": 8,
            "imbalance_ratio": 1.0,
            "image_size": 64,
            "noise_mode": "family",
            "noise_rate": 0.2,
            "exact_duplicate_count": 0,
            "near_duplicate_count": 0,
            "corrupt_count": 0,
            "truncated_count": 0,
        },
        "dataset": {
            "stage": "synthetic",
            "dataset_root": str(dataset_root),
            "class_mapping": "metadata/classes.csv",
            "class_mapping_provenance": "official_file",
            "extensions": [".png"],
            "allow_truncated_images": False,
            "sources": {
                "train": {
                    "root": "images",
                    "label_source": "manifest",
                    "manifest": "metadata/samples.csv",
                }
            },
            "audit": {
                "output_root": str(audit_root),
                "workers": 2,
                "near_duplicate_hamming": 3,
                "near_duplicate_max_bucket": 500,
                "fail_on_bad_files": True,
                "fail_on_label_conflicts": True,
            },
            "split": {
                "validation_fraction": 0.25,
                "seed": 20260920,
                "min_train_per_class": 2,
            },
            "transforms": {
                "image_size": 224,
                "resize_size": 224,
                "train_crop_scale_min": 0.8,
                "train_horizontal_flip_probability": 0.5,
            },
            "submission": {
                "filename_mode": "relative_path",
                "official_class_id_pattern": "\\d{4}",
            },
        },
    }
    config_path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )

    started = time.perf_counter()
    synthetic = synthetic_config_from_mapping(document, config_path.parent)
    generation = generate_synthetic_dataset(synthetic)
    dataset_config = load_dataset_config(config_path)
    audit = write_audit_report(dataset_config, "train")
    manifest_path = audit_root / "train" / "samples.csv"
    records = read_manifest(manifest_path)
    assignment = build_grouped_stratified_split(
        records,
        validation_fraction=dataset_config.split.validation_fraction,
        seed=dataset_config.split.seed,
        min_train_per_class=dataset_config.split.min_train_per_class,
        allowed_decode_statuses=frozenset({"ok"}),
    )
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_digest = write_split(assignment, split_path)

    command = [
        str(args.python.resolve()),
        "scripts/smoke_peft.py",
        "--dataset-config",
        str(config_path),
        "--model-config",
        str(ROOT / "configs/model/openai_clip_vit_b32.yaml"),
        "--manifest",
        str(manifest_path),
        "--split",
        str(split_path),
        "--mode",
        "last_block",
        "--output",
        str(output_dir),
        "--max-train",
        "14",
        "--max-validation",
        "14",
        "--batch-size",
        "2",
        "--steps",
        "1",
        "--device",
        args.device,
        "--build-local-package",
    ]
    return_code = _run_smoke(command, output_dir, log_path)
    if return_code != 0:
        raise RuntimeError(f"PEFT migration smoke failed; inspect {log_path}")
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    summary = {
        "schema": "aictrack1.stage2_migration_smoke.v1",
        "migration_smoke_only": True,
        "competition_test_accessed": False,
        "run_root": str(run_root),
        "elapsed_seconds": time.perf_counter() - started,
        "generation": generation,
        "audit": audit,
        "manifest_sha256": _sha256(manifest_path),
        "split_sha256": split_digest,
        "checkpoint_sha256": result["checkpoint_sha256"],
        "checkpoint_restore_exact": True,
        "local_validation_package": result["local_validation_package"],
        "training_lock_released": not (ROOT / DEFAULT_LOCK_RELATIVE_PATH).exists(),
        "command": subprocess.list2cmdline(command),
        "pid": os.getpid(),
    }
    summary_path = run_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
