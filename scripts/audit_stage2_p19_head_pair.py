"""Paired P19 head screen on one Stage2 train-derived validation fold only."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.class_mapping import ClassMapping  # noqa: E402
from aictrack1.data.contracts import load_dataset_config  # noqa: E402
from aictrack1.data.feature_cache import load_feature_cache  # noqa: E402
from aictrack1.data.manifest import sha256_file  # noqa: E402
from aictrack1.engine.provenance import CheckpointProvenance  # noqa: E402
from aictrack1.evaluation.balanced_prior import fit_uniform_soft_bias  # noqa: E402
from aictrack1.evaluation.feature_geometry import grouped_bootstrap_delta  # noqa: E402
from aictrack1.models.heads import load_classifier_head_checkpoint  # noqa: E402


def _groups_and_conflicts(manifest: Path) -> tuple[dict[str, str], set[str]]:
    groups: dict[str, str] = {}
    labels_by_hash: dict[str, set[str]] = {}
    hash_by_id: dict[str, str] = {}
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "source_split", "near_duplicate_group", "sha256", "official_class_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Stage2 train manifest columns are incomplete")
        for row in reader:
            if row["source_split"] != "train" or row["sample_id"] in groups:
                raise ValueError("manifest contains non-train or duplicate sample")
            sample_id = row["sample_id"]
            groups[sample_id] = (
                f"near:{row['near_duplicate_group']}"
                if row["near_duplicate_group"] else f"singleton:{sample_id}"
            )
            hash_by_id[sample_id] = row["sha256"]
            labels_by_hash.setdefault(row["sha256"], set()).add(row["official_class_id"])
    conflicts = {sample_id for sample_id, digest in hash_by_id.items() if len(labels_by_hash[digest]) > 1}
    return groups, conflicts


@torch.inference_mode()
def _logits(head: torch.nn.Module, features: torch.Tensor, device: torch.device) -> torch.Tensor:
    head = head.to(device).eval()
    return torch.cat([
        head(chunk.to(device).float()).float().cpu()
        for chunk in features.split(4096)
    ])


def _metrics(predictions: torch.Tensor, targets: torch.Tensor, tail_classes: torch.Tensor,
             ordinary: torch.Tensor, num_classes: int) -> dict[str, float | int]:
    correct = predictions.eq(targets)
    totals = torch.bincount(targets, minlength=num_classes)
    class_correct = torch.bincount(targets[correct], minlength=num_classes)
    if bool((totals == 0).any()) or not ordinary.any():
        raise ValueError("validation must cover every class and ordinary samples")
    recall = class_correct.float() / totals.float()
    return {
        "top1": float(correct.float().mean()),
        "macro": float(recall.mean()),
        "tail_class_macro": float(recall[tail_classes].mean()),
        "ordinary_top1": float(correct[ordinary].float().mean()),
        "zero_recall_classes": int((class_correct == 0).sum()),
    }


def _group_sufficient_statistics(
    group_ids: list[str], baseline: np.ndarray, candidate: np.ndarray
) -> list[dict[str, str | int]]:
    grouped: dict[str, list[int]] = {}
    for group_id, first, second in zip(group_ids, baseline, candidate, strict=True):
        values = grouped.setdefault(group_id, [0, 0, 0])
        values[0] += 1
        values[1] += int(first)
        values[2] += int(second)
    return [
        {"group_id": group_id, "count": values[0],
         "baseline_correct": values[1], "candidate_correct": values[2]}
        for group_id, values in grouped.items()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(2, 3), required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "artifacts" / "reports").resolve()) or output.exists():
        raise ValueError("output must be a new path under artifacts/reports")
    baseline_config = ROOT / "configs" / "experiment" / f"stage2_fold{args.fold}_c5_h1_336_seed20260921.yaml"
    raw = yaml.safe_load(baseline_config.read_text(encoding="utf-8"))["experiment"]
    data_config = load_dataset_config((baseline_config.parent / raw["dataset_config"]).resolve())
    if data_config.competition_stage != "stage2":
        raise ValueError("P19 must use Stage2")
    mapping = ClassMapping.from_csv(data_config.class_mapping)
    cache_path = (baseline_config.parent / raw["feature_cache"]).resolve()
    cache = load_feature_cache(cache_path)
    manifest = ROOT / "artifacts" / "audits" / "stage2" / "train" / "samples.csv"
    if cache.metadata.source != "train" or len(cache.sample_ids) != 148695:
        raise ValueError("P19 requires complete Stage2 train-only features")
    if cache.metadata.manifest_sha256 != sha256_file(manifest):
        raise ValueError("feature cache manifest SHA mismatch")
    if cache.metadata.class_mapping_sha256 != mapping.digest() or len(mapping) != 750:
        raise ValueError("feature cache class mapping mismatch")
    if cache.metadata.peft_checkpoint_sha256 is not None:
        raise ValueError("P19 H1 screen requires original OpenAI CLIP features")
    train_indices = torch.tensor([index for index, part in enumerate(cache.partitions) if part == "train"])
    val_indices = torch.tensor([index for index, part in enumerate(cache.partitions) if part == "validation"])
    if len(train_indices) != 118958 or len(val_indices) != 29737:
        raise ValueError("P19 fold train/validation partition size mismatch")
    counts = torch.bincount(cache.labels.index_select(0, train_indices).long(), minlength=750)
    if bool((counts <= 0).any()):
        raise ValueError("P19 fold has a missing training class")
    groups_by_id, conflicts = _groups_and_conflicts(manifest)
    sample_ids = [cache.sample_ids[index] for index in val_indices.tolist()]
    if len(groups_by_id) != 148695 or any(sample_id not in groups_by_id for sample_id in sample_ids):
        raise ValueError("manifest sample IDs differ from the cache")
    group_ids = [groups_by_id[sample_id] for sample_id in sample_ids]
    ordinary = torch.tensor([sample_id not in conflicts for sample_id in sample_ids], dtype=torch.bool)
    targets = cache.labels.index_select(0, val_indices).long()
    features = cache.features.index_select(0, val_indices).float()
    tail_classes = torch.argsort(counts, stable=True)[:375]
    provenance = CheckpointProvenance(
        competition_stage="stage2",
        class_mapping_sha256=mapping.digest(),
        train_manifest_sha256=cache.metadata.manifest_sha256,
        split_sha256=cache.metadata.split_sha256,
        clip_weight_sha256=cache.metadata.clip_weight_sha256,
        num_classes=750,
    )
    baseline_path = (baseline_config.parent / raw["output_dir"] / "best.pt").resolve()
    candidate_path = args.candidate.resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    results = {}
    corrected = {}
    raw_predictions = {}
    for name, path in (("baseline", baseline_path), ("candidate", candidate_path)):
        head, checkpoint = load_classifier_head_checkpoint(
            path, feature_dim=cache.metadata.feature_dim, num_classes=750,
            expected_clip_weight_sha256=cache.metadata.clip_weight_sha256,
            expected_class_mapping_sha256=mapping.digest(),
            expected_split_sha256=cache.metadata.split_sha256,
            expected_provenance=provenance, forbid_full_train_initializer=True,
        )
        actual_kind = str(checkpoint.get("loss", {}).get("kind", "cross_entropy"))
        expected_kind = "cross_entropy" if name == "baseline" else "balanced_softmax"
        if actual_kind != expected_kind:
            raise ValueError(f"{name} loss is not {expected_kind}")
        if name == "candidate" and checkpoint["feature_cache_metadata"].get("training_class_support") != counts.tolist():
            raise ValueError("candidate class counts differ from the train partition")
        scores = _logits(head, features, device)
        raw_predictions[name] = scores.argmax(dim=1)
        bias, bias_details = fit_uniform_soft_bias(
            scores.to(device), temperature_multiplier=0.5, iterations=60
        )
        corrected[name] = (scores.to(device) + 0.75 * bias).argmax(dim=1).cpu()
        results[name] = {
            "checkpoint_sha256": sha256_file(path),
            "raw_center": _metrics(raw_predictions[name], targets, tail_classes, ordinary, 750),
            "p7_frozen_prior_center": _metrics(corrected[name], targets, tail_classes, ordinary, 750),
            "prior_fit_unlabeled_diagnostics": bias_details,
        }
    first = corrected["baseline"].eq(targets).float().numpy().astype(np.float64)
    second = corrected["candidate"].eq(targets).float().numpy().astype(np.float64)
    bootstrap = grouped_bootstrap_delta(first, second, group_ids, seed=20260925, replicates=500)
    raw_first = raw_predictions["baseline"].eq(targets).float().numpy().astype(np.float64)
    raw_second = raw_predictions["candidate"].eq(targets).float().numpy().astype(np.float64)
    raw_bootstrap = grouped_bootstrap_delta(
        raw_first, raw_second, group_ids, seed=20260925, replicates=500
    )
    report = {
        "scope": "Stage2 train-derived fold validation only; original CLIP center features; no test",
        "fold": args.fold,
        "feature_cache": str(cache_path),
        "manifest_sha256": sha256_file(manifest),
        "split_sha256": cache.metadata.split_sha256,
        "class_mapping_sha256": mapping.digest(),
        "validation_count": len(val_indices),
        "prior_recipe": "uniform_soft_t0.5_s0.75 fitted to each head's unlabeled validation logits",
        "models": results,
        "raw_center_paired_near_duplicate_group_bootstrap": raw_bootstrap,
        "raw_center_group_sufficient_statistics": _group_sufficient_statistics(
            group_ids, raw_first, raw_second
        ),
        "corrected_paired_near_duplicate_group_bootstrap": bootstrap,
        "caveat": "raw center-feature head screen is the gate; corrected long-tail-fold view is descriptive only, not the P7 balanced-proxy procedure, four-view PEFT, or an estimate of hidden test score",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"fold": args.fold, "bootstrap": bootstrap}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
