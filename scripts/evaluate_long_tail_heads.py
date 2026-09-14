"""Evaluate frozen-feature heads on fixed validation and the long-tail stress partitions.

Reads only the current-stage training feature cache and the deterministic
long-tail split derived from the frozen train partition. It never reads
competition test data and never writes a submission package.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.class_mapping import ClassMapping  # noqa: E402
from aictrack1.data.contracts import load_dataset_config  # noqa: E402
from aictrack1.data.feature_cache import load_feature_cache  # noqa: E402
from aictrack1.data.long_tail_split import (  # noqa: E402
    LONG_TAIL_EVAL,
    LONG_TAIL_TRAIN,
    read_split_assignments,
    select_partition_indices,
    summarize_partition,
)
from aictrack1.data.manifest import read_manifest  # noqa: E402
from aictrack1.evaluation.classification import classification_diagnostics  # noqa: E402
from aictrack1.evaluation.long_tail import (  # noqa: E402
    long_tail_diagnostics,
    read_stratification,
    support_stratified_accuracy,
)
from aictrack1.models.heads import build_classifier_head  # noqa: E402
from aictrack1.models.single_model import dump_json, resolve_relative, sha256_file  # noqa: E402


def _assert_not_test(path: Path, *, label: str) -> None:
    if "test" in [part.lower() for part in path.resolve().parts]:
        raise ValueError(f"{label} resolves inside a test directory: {path}")


@torch.inference_mode()
def _logits_for(model, features: torch.Tensor, *, device: torch.device, batch_size: int) -> torch.Tensor:
    model.eval()
    chunks = []
    for (batch,) in DataLoader(TensorDataset(features), batch_size=batch_size):
        chunks.append(model(batch.to(device)).float().cpu())
    return torch.cat(chunks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-tail-config", required=True, type=Path)
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="LABEL=PATH checkpoint to evaluate; repeatable",
    )
    args = parser.parse_args()
    if not args.checkpoint:
        raise ValueError("at least one --checkpoint LABEL=PATH is required")

    long_tail_config_path = args.long_tail_config.resolve()
    long_tail_config = yaml.safe_load(long_tail_config_path.read_text(encoding="utf-8"))[
        "long_tail_split"
    ]
    outputs = long_tail_config["outputs"]
    split_path = resolve_relative(long_tail_config_path.parent, outputs["split"])
    strata_path = resolve_relative(long_tail_config_path.parent, outputs["stratification"])
    summary_path = resolve_relative(long_tail_config_path.parent, outputs["summary"])
    for path, label in ((split_path, "long-tail split"), (strata_path, "stratification")):
        _assert_not_test(path, label=label)
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}")

    dataset_config = load_dataset_config(args.dataset_config.resolve())
    manifest_path = args.manifest.resolve()
    frozen_split_path = args.split.resolve()
    _assert_not_test(manifest_path, label="manifest")
    _assert_not_test(frozen_split_path, label="frozen split")
    mapping = ClassMapping.from_csv(dataset_config.class_mapping)
    num_classes = len(mapping)
    strata = read_stratification(strata_path)
    if strata.num_classes != num_classes:
        raise ValueError("stratification and class mapping disagree")

    all_records = read_manifest(manifest_path)
    frozen_assignments = read_split_assignments(frozen_split_path)
    long_tail_assignment = read_split_assignments(split_path, column="long_tail_split")
    reference_counts = summarize_partition(
        all_records, frozen_assignments, partition="train", num_classes=num_classes
    )
    kept_counts = summarize_partition(
        all_records, long_tail_assignment, partition=LONG_TAIL_TRAIN, num_classes=num_classes
    )
    long_tail_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    cache = load_feature_cache(args.feature_cache.resolve())
    if cache.metadata.source != "train":
        raise ValueError("stress diagnostics refuse test feature caches")
    sample_ids = list(cache.sample_ids)
    sample_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}

    validation_positions = [
        index
        for index, partition in enumerate(cache.partitions)
        if partition == "validation"
    ]
    long_tail_eval_positions = [
        sample_index[sample_id]
        for sample_id in sample_ids
        if long_tail_assignment.get(sample_id) == LONG_TAIL_EVAL
    ]
    if not validation_positions or not long_tail_eval_positions:
        raise ValueError("evaluation partitions are empty")

    partitions = {
        "fixed_noisy_validation": validation_positions,
        "long_tail_eval": long_tail_eval_positions,
    }
    selected_train_positions = select_partition_indices(
        sample_ids,
        long_tail_assignment,
        partition=LONG_TAIL_TRAIN,
        available=set(sample_ids),
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    report: dict = {
        "scope": (
            "current-stage training feature cache and deterministic long-tail stress split "
            "only; competition test data not accessed"
        ),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "num_classes": num_classes,
        "long_tail_design": {
            "seed": long_tail_summary["seed"],
            "schedule": long_tail_summary["schedule"],
            "stratification": long_tail_summary["stratification"],
            "partitions": long_tail_summary["partitions"],
            "duplicate_cluster_leakage_count": long_tail_summary["duplicate_cluster_leakage_count"],
            "retention": long_tail_summary["retention"],
            "split_sha256": long_tail_summary["hashes"]["long_tail_split"],
        },
        "partitions_evaluated": {
            "fixed_noisy_validation": len(validation_positions),
            "long_tail_eval": len(long_tail_eval_positions),
            "long_tail_train": len(selected_train_positions),
        },
        "models": {},
    }
    for specification in args.checkpoint:
        if "=" not in specification:
            raise ValueError(f"--checkpoint must be LABEL=PATH, got {specification!r}")
        label, raw_path = specification.split("=", 1)
        checkpoint_path = Path(raw_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = (ROOT / checkpoint_path).resolve()
        _assert_not_test(checkpoint_path, label=f"checkpoint {label}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("feature_cache_metadata", {}).get("manifest_sha256") != cache.metadata.manifest_sha256:
            raise ValueError(f"checkpoint {label} belongs to a different manifest")
        head_config = checkpoint["head"]
        loss_config = checkpoint.get("loss", {"kind": "cross_entropy"})
        model = build_classifier_head(
            str(head_config["kind"]),
            cache.metadata.feature_dim,
            num_classes,
            cosine_initial_scale=float(head_config.get("initial_scale", 10.0)),
            cosine_learnable_scale=bool(head_config.get("learnable_scale", True)),
        )
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        metadata = checkpoint.get("feature_cache_metadata", {})
        training_scope = str(metadata.get("training_scope", "unknown"))
        stored_support = metadata.get("training_class_support")
        if stored_support is not None:
            starvation_support = [int(value) for value in stored_support]
            if len(starvation_support) != num_classes:
                raise ValueError(f"checkpoint {label} stores a partial class support vector")
        elif training_scope in {"long_tail_stress", "balanced_size_matched_control"}:
            starvation_support = summarize_partition(
                all_records,
                long_tail_assignment,
                partition=LONG_TAIL_TRAIN,
                num_classes=num_classes,
            )
        else:
            starvation_support = summarize_partition(
                all_records,
                frozen_assignments,
                partition="train",
                num_classes=num_classes,
            )
        entry: dict = {
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "training_scope": training_scope,
            "training_record_count": metadata.get("training_record_count"),
            "selection_rule": checkpoint.get("selection_rule"),
            "training_class_support_stratum": training_scope,
        }
        for partition_name, positions in partitions.items():
            features = cache.features.index_select(
                0, torch.tensor(positions, dtype=torch.long)
            ).float()
            targets = cache.labels.index_select(0, torch.tensor(positions, dtype=torch.long))
            logits = _logits_for(model, features, device=device, batch_size=args.batch_size)
            predictions = logits.argmax(dim=1)
            entry[partition_name] = {
                "classification": classification_diagnostics(
                    predictions, targets, num_classes=num_classes
                ),
                "long_tail": long_tail_diagnostics(
                    predictions,
                    targets,
                    num_classes=num_classes,
                    stratification=strata,
                    reference_class_counts=reference_counts,
                    evaluated_class_counts=kept_counts,
                ),
                "starvation_probe": support_stratified_accuracy(
                    predictions,
                    targets,
                    num_classes=num_classes,
                    train_class_counts=starvation_support,
                ),
            }
            if loss_config.get("kind") == "logit_clip_cross_entropy":
                entry[partition_name]["logit_clip"] = {
                    "tau": float(loss_config["logit_clip_tau"]),
                    "p_norm": float(loss_config["logit_clip_p_norm"]),
                }
        report["models"][label] = entry

    output_path = args.output.resolve()
    dump_json(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
