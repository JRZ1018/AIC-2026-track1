"""Train one reproducible single-model PEFT candidate from the A1 cosine head."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.class_mapping import ClassMapping
from aictrack1.data.contracts import (
    ensure_output_path_is_safe,
    load_dataset_config,
    validate_experiment_runtime_contract,
)
from aictrack1.data.feature_cache import load_feature_cache
from aictrack1.data.indexed_images import IndexedImage, IndexedImageDataset, read_indexed_images
from aictrack1.data.transforms import build_split_transforms
from aictrack1.engine.peft import load_trainable_state_dict, trainable_state_dict
from aictrack1.engine.ema import TrainableEMA, optimizer_step_completed
from aictrack1.engine.environment_identity import (
    capture_environment_identity,
    environment_hash,
)
from aictrack1.engine.provenance import (
    CheckpointProvenance,
    validate_checkpoint_provenance,
)
from aictrack1.evaluation.classification import calibration_diagnostics, classification_diagnostics
from aictrack1.evaluation.long_tail import (
    Stratification,
    long_tail_diagnostics,
    stratify_classes,
)
from aictrack1.methods.confidence import (
    ClasswiseHistoryWeighting,
    visual_prototype_soft_weights,
    weighted_cross_entropy,
)
from aictrack1.methods.conflicts import conflict_group_weight_lookup
from aictrack1.methods.elr import ELRLoss, ELRState
from aictrack1.methods.gce import generalized_cross_entropy
from aictrack1.methods.nce_rce import (
    normalized_cross_entropy_plus_reverse_cross_entropy,
    validate_nce_rce_parameters,
)
from aictrack1.methods.sce import symmetric_cross_entropy, validate_sce_parameters
from aictrack1.methods.long_tail_training import (
    mild_class_balanced_sample_weights,
    train_prior_logit_adjustment,
)
from aictrack1.models.clip_vit_b32 import load_clip_vit_b32
from aictrack1.models.clip_resolution import enable_interpolated_visual_resolution
from aictrack1.models.heads import load_classifier_head_checkpoint
from aictrack1.models.peft import PEFTImageClassifier, configure_visual_peft


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    """Return a verified commit or an explicit archive sentinel.

    Compute-only hosts may receive a verified source archive without ``.git``.
    In that case a valid ``AICTRACK1_GIT_COMMIT`` may identify the source
    checkout. If neither is available, the archive is recorded as
    ``unknown_archive``; the launch receipt must then carry the file-level
    source hashes. A missing Git checkout is provenance metadata, not a reason
    to fail after a fully completed training run.
    """

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        archived_commit = os.environ.get("AICTRACK1_GIT_COMMIT", "").strip().lower()
        if len(archived_commit) != 40 or any(
            character not in "0123456789abcdef" for character in archived_commit
        ):
            return "unknown_archive"
        return archived_commit


def _base_loss_per_sample(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    kind: str,
    q: float,
    label_smoothing: float,
    sce_alpha: float = 0.1,
    sce_beta: float = 1.0,
    sce_log_zero: float = -4.0,
    nce_alpha: float = 750.0,
    nce_beta: float = 0.1,
    nce_log_zero: float = -4.0,
    elr_loss: ELRLoss | None = None,
    sample_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the configured base loss for each sample.

    The AC/confidence update below intentionally keeps using detached CE losses;
    this function only changes the gradient-bearing loss used for the optimizer.
    """
    if kind == "gce":
        return generalized_cross_entropy(logits, targets, q=q, reduction="none")
    if kind == "sce":
        if label_smoothing != 0.0:
            raise ValueError("SCE and label smoothing are separate ablations")
        return symmetric_cross_entropy(
            logits, targets, alpha=sce_alpha, beta=sce_beta,
            log_zero=sce_log_zero, reduction="none",
        )
    if kind == "nce_rce":
        return normalized_cross_entropy_plus_reverse_cross_entropy(
            logits, targets, alpha=nce_alpha, beta=nce_beta,
            log_zero=nce_log_zero, reduction="none",
        )
    if kind == "elr":
        if elr_loss is None or sample_indices is None:
            raise ValueError("ELR requires its stateful loss and sample indices")
        return elr_loss(logits, targets, sample_indices, reduction="none")
    if kind != "cross_entropy":
        raise ValueError(f"unsupported base_loss kind {kind!r}")
    return F.cross_entropy(
        logits,
        targets,
        reduction="none",
        label_smoothing=label_smoothing,
    )


def _weighted_base_loss(
    per_sample_loss: torch.Tensor,
    weights: torch.Tensor | None,
) -> torch.Tensor:
    if weights is None:
        return per_sample_loss.mean()
    normalized = weights.to(device=per_sample_loss.device, dtype=per_sample_loss.dtype)
    if normalized.ndim != 1 or normalized.shape != per_sample_loss.shape:
        raise ValueError("weights must match per-sample loss shape")
    if not torch.isfinite(normalized).all() or bool((normalized <= 0).any()):
        raise ValueError("weights must be finite and strictly positive")
    return (per_sample_loss * normalized).sum() / normalized.sum()


def _save_atomic(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _model_settings(path: Path) -> tuple[Path, str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = raw.get("model", raw)
    if raw.get("provider") != "OpenAI" or raw.get("name") != "ViT-B/32":
        raise ValueError("model config must declare OpenAI ViT-B/32")
    if raw.get("runtime_downloads_allowed") is not False:
        raise ValueError("runtime downloads must be disabled")
    return _resolve(path.parent, str(raw["weight_path"])), str(raw["expected_weight_sha256"])


def _conflict_ids(manifest: Path, conflicts: Path) -> set[str]:
    with conflicts.open("r", encoding="utf-8-sig", newline="") as handle:
        hashes = {str(row["sha256"]) for row in csv.DictReader(handle)}
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            str(row["sample_id"])
            for row in csv.DictReader(handle)
            if str(row["sha256"]) in hashes
        }


def _loader(
    dataset: IndexedImageDataset,
    *,
    batch_size: int,
    workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    shuffle: bool,
    seed: int,
    device: torch.device,
    sample_weights: torch.Tensor | None = None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if sample_weights is not None:
        if not shuffle:
            raise ValueError("sample weights are only valid for a shuffled training loader")
        if sample_weights.ndim != 1 or sample_weights.numel() != len(dataset):
            raise ValueError("sample weights must match the training dataset")
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle and sampler is None,
        "sampler": sampler,
        "generator": torch.Generator().manual_seed(seed + 1),
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        loader_kwargs.update(
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
    return DataLoader(**loader_kwargs)


def _data_loader_settings(training: dict[str, Any]) -> tuple[int, int, bool]:
    workers = int(training.get("workers", 0))
    prefetch_factor = int(training.get("prefetch_factor", 2))
    persistent_workers_raw = training.get("persistent_workers", True)
    if not isinstance(persistent_workers_raw, bool):
        raise ValueError("persistent_workers must be a boolean")
    if workers < 0 or prefetch_factor < 1:
        raise ValueError("workers must be non-negative and prefetch_factor must be positive")
    return workers, prefetch_factor, persistent_workers_raw


def _runtime_settings(raw: dict[str, Any]) -> dict[str, bool]:
    """Read opt-in runtime switches without changing algorithm defaults.

    ``defer_epoch_metrics`` moves diagnostic reductions from every batch to a
    single reduction at epoch end. It does not change logits, losses, gradients,
    sampling, or checkpoint selection.
    """

    runtime_raw = raw.get("runtime", {})
    if runtime_raw is None:
        runtime_raw = {}
    if not isinstance(runtime_raw, dict):
        raise ValueError("runtime must be a mapping when provided")
    defer_epoch_metrics = runtime_raw.get("defer_epoch_metrics", False)
    if not isinstance(defer_epoch_metrics, bool):
        raise ValueError("runtime.defer_epoch_metrics must be a boolean")
    reuse_validation_loader = runtime_raw.get("reuse_validation_loader", False)
    if not isinstance(reuse_validation_loader, bool):
        raise ValueError("runtime.reuse_validation_loader must be a boolean")
    return {
        "defer_epoch_metrics": defer_epoch_metrics,
        "reuse_validation_loader": reuse_validation_loader,
    }


@torch.inference_mode()
def _evaluate(
    model: PEFTImageClassifier,
    records: list[IndexedImage],
    dataset: IndexedImageDataset | None,
    *,
    loader: DataLoader | None = None,
    batch_size: int,
    workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    device: torch.device,
    num_classes: int,
    conflict_ids: set[str],
    stratification: Stratification,
) -> dict[str, Any]:
    model.eval()
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    if loader is None:
        if dataset is None:
            raise ValueError("evaluation requires either a dataset or a loader")
        loader = _loader(
            dataset,
            batch_size=batch_size,
            workers=workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            shuffle=False,
            seed=0,
            device=device,
        )
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits.append(model(images).float().cpu())
        targets.append(batch["label"])
    all_logits = torch.cat(logits)
    all_targets = torch.cat(targets).long()
    predictions = all_logits.argmax(dim=1)
    conflict_mask = torch.tensor([record.sample_id in conflict_ids for record in records])
    return {
        "all": classification_diagnostics(predictions, all_targets, num_classes=num_classes),
        "calibration": calibration_diagnostics(all_logits, all_targets),
        "cross_label_exact_conflicts": classification_diagnostics(
            predictions[conflict_mask], all_targets[conflict_mask], num_classes=num_classes
        ),
        "remaining": classification_diagnostics(
            predictions[~conflict_mask], all_targets[~conflict_mask], num_classes=num_classes
        ),
        "conflict_sample_count": int(conflict_mask.sum()),
        "long_tail": long_tail_diagnostics(
            predictions,
            all_targets,
            num_classes=num_classes,
            stratification=stratification,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    raw_document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw = raw_document.get("experiment", raw_document)
    if raw.get("execution_enabled", True) is not True:
        raise RuntimeError("this PEFT configuration is design-only and not approved for execution")
    base = config_path.parent
    dataset_path = _resolve(base, str(raw["dataset_config"]))
    model_path = _resolve(base, str(raw["model_config"]))
    manifest_path = _resolve(base, str(raw["manifest"]))
    split_path = _resolve(base, str(raw["split"]))
    conflicts_path = _resolve(base, str(raw["label_conflicts"]))
    head_checkpoint_setting = raw.get("head_checkpoint", raw.get("a1_head_checkpoint"))
    if head_checkpoint_setting is None:
        raise ValueError("PEFT config must define head_checkpoint or legacy a1_head_checkpoint")
    if "head_checkpoint" in raw and "a1_head_checkpoint" in raw:
        raise ValueError("PEFT config must not define both head checkpoint fields")
    head_checkpoint_path = _resolve(base, str(head_checkpoint_setting))
    output = _resolve(base, str(raw["output_dir"]))
    dataset_config = load_dataset_config(dataset_path)
    competition_stage = validate_experiment_runtime_contract(raw, dataset_config)
    ensure_output_path_is_safe(dataset_config, output)
    if output.exists() and args.resume is None:
        raise FileExistsError(f"PEFT output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    training = raw["training"]
    ema_raw = raw.get("trainable_ema")
    if ema_raw is not None and not isinstance(ema_raw, dict):
        raise ValueError("trainable_ema must be a mapping")
    ema_enabled = bool(ema_raw and ema_raw.get("enabled", False))
    ema_decay = float(ema_raw.get("decay", 0.999)) if ema_enabled else None
    training_scope = str(raw.get("training_scope", "fixed_split_validation"))
    if training_scope not in {"fixed_split_validation", "full_train_fixed_epochs"}:
        raise ValueError(f"unsupported PEFT training scope {training_scope!r}")
    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    workers, prefetch_factor, persistent_workers = _data_loader_settings(training)
    runtime_settings = _runtime_settings(raw)
    defer_epoch_metrics = runtime_settings["defer_epoch_metrics"]
    reuse_validation_loader = runtime_settings["reuse_validation_loader"]
    seed = int(training["seed"])
    label_smoothing = float(training.get("label_smoothing", 0.0))
    base_loss_raw = raw.get("base_loss", {"kind": "cross_entropy"})
    if base_loss_raw is None:
        base_loss_raw = {"kind": "cross_entropy"}
    if not isinstance(base_loss_raw, dict):
        raise ValueError("base_loss must be a mapping when provided")
    base_loss_kind = str(base_loss_raw.get("kind", "cross_entropy"))
    if base_loss_kind not in {"cross_entropy", "gce", "sce", "nce_rce", "elr"}:
        raise ValueError(f"unsupported base_loss kind {base_loss_kind!r}")
    base_loss_q = float(base_loss_raw.get("q", 0.7))
    if base_loss_kind == "gce" and not 0.0 < base_loss_q <= 1.0:
        raise ValueError("GCE q must be in (0, 1]")
    sce_alpha = float(base_loss_raw.get("alpha", 0.1))
    sce_beta = float(base_loss_raw.get("beta", 1.0))
    sce_log_zero = float(base_loss_raw.get("log_zero", -4.0))
    if base_loss_kind == "sce":
        validate_sce_parameters(sce_alpha, sce_beta, sce_log_zero)
    nce_alpha = float(base_loss_raw.get("alpha", 750.0))
    nce_beta = float(base_loss_raw.get("beta", 0.1))
    nce_log_zero = float(base_loss_raw.get("log_zero", -4.0))
    if base_loss_kind == "nce_rce":
        validate_nce_rce_parameters(nce_alpha, nce_beta, nce_log_zero)
    elr_beta = float(base_loss_raw.get("beta", 0.7))
    elr_lambda = float(base_loss_raw.get("lambda", 3.0))
    elr_eps = float(base_loss_raw.get("eps", 1e-6))
    if base_loss_kind == "elr":
        if not 0.0 <= elr_beta < 1.0 or elr_lambda < 0.0 or not 0.0 < elr_eps < 0.5:
            raise ValueError("invalid ELR beta, lambda or eps")
    if epochs < 1 or batch_size < 1:
        raise ValueError("invalid epoch or batch-size configuration")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    if base_loss_kind == "gce" and label_smoothing > 0.0:
        raise ValueError("GCE and label smoothing are separate ablations")
    if base_loss_kind == "sce" and label_smoothing > 0.0:
        raise ValueError("SCE and label smoothing are separate ablations")
    if base_loss_kind == "nce_rce" and label_smoothing > 0.0:
        raise ValueError("NCE+RCE and label smoothing are separate ablations")
    if base_loss_kind == "elr" and label_smoothing > 0.0:
        raise ValueError("ELR and label smoothing are separate ablations")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(str(raw.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    records = read_indexed_images(
        dataset_config,
        source="train",
        manifest_path=manifest_path,
        split_path=split_path,
    )
    validate_experiment_runtime_contract(
        raw,
        dataset_config,
        train_count=len(records),
    )
    if training_scope == "full_train_fixed_epochs":
        expected_value = raw.get(
            "expected_train_count", dataset_config.expected_train_count
        )
        if expected_value is None:
            raise ValueError("full-train run requires expected_train_count")
        expected_train_count = int(expected_value)
        if len(records) != expected_train_count:
            raise ValueError("full-train manifest count differs from expected_train_count")
        if {record.partition for record in records} != {"train", "validation"}:
            raise ValueError("full-train manifest has unexpected partitions")
        train_records = records
        validation_records = []
    else:
        train_records = [record for record in records if record.partition == "train"]
        validation_records = [record for record in records if record.partition == "validation"]
    transforms = build_split_transforms(dataset_config.transforms)
    train_dataset = IndexedImageDataset(
        train_records,
        transforms["train"],
        allow_truncated_images=dataset_config.allow_truncated_images,
        stochastic_seed=seed,
    )
    validation_dataset = (
        None
        if training_scope == "full_train_fixed_epochs"
        else IndexedImageDataset(
            validation_records,
            transforms["validation"],
            allow_truncated_images=dataset_config.allow_truncated_images,
        )
    )
    validation_loader: DataLoader | None = None
    if validation_dataset is not None and reuse_validation_loader:
        validation_loader = _loader(
            validation_dataset,
            batch_size=batch_size,
            workers=workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            shuffle=False,
            seed=0,
            device=device,
        )
    confidence_raw = raw.get("confidence_weighting")
    visual_trust_raw = raw.get("visual_prototype_weighting")
    if confidence_raw is not None and visual_trust_raw is not None:
        raise ValueError("dynamic confidence and visual prototype weighting are separate ablations")
    if label_smoothing > 0.0 and (confidence_raw is not None or visual_trust_raw is not None):
        raise ValueError("label smoothing and sample weighting are separate ablations")
    confidence: ClasswiseHistoryWeighting | None = None
    confidence_warmup_epochs = 0
    if confidence_raw is not None and bool(confidence_raw.get("enabled", True)):
        confidence_warmup_epochs = int(confidence_raw.get("warmup_epochs", 1))
        if confidence_warmup_epochs < 1 or confidence_warmup_epochs >= epochs:
            raise ValueError("confidence warmup_epochs must be in [1, epochs)")
        confidence = ClasswiseHistoryWeighting(
            torch.tensor([record.sample_index for record in train_records]),
            torch.tensor([int(record.internal_class_index) for record in train_records]),
            momentum=float(confidence_raw.get("momentum", 0.8)),
            history_mix=float(confidence_raw.get("history_mix", 0.5)),
            min_weight=float(confidence_raw.get("min_weight", 0.2)),
            magnitude_mix=float(confidence_raw.get("magnitude_mix", 0.0)),
            tail_class_fraction=(
                None
                if confidence_raw.get("tail_class_fraction") is None
                else float(confidence_raw["tail_class_fraction"])
            ),
            tail_min_mean_weight=(
                None
                if confidence_raw.get("tail_min_mean_weight") is None
                else float(confidence_raw["tail_min_mean_weight"])
            ),
            min_effective_mass_per_class=(
                None
                if confidence_raw.get("min_effective_mass_per_class") is None
                else float(confidence_raw["min_effective_mass_per_class"])
            ),
        )
    conflict_raw = raw.get("conflict_loss", {"enabled": False})
    if conflict_raw is None:
        conflict_raw = {"enabled": False}
    if not isinstance(conflict_raw, dict):
        raise ValueError("conflict_loss must be a mapping when provided")
    conflict_enabled = bool(conflict_raw.get("enabled", False))
    conflict_mode = str(conflict_raw.get("mode", "group_normalized"))
    if conflict_enabled and conflict_mode != "group_normalized":
        raise ValueError("unsupported conflict_loss mode")
    conflict_weight_lookup: torch.Tensor | None = None
    conflict_diagnostics: dict[str, int | str] | None = None
    if conflict_enabled:
        conflict_weight_lookup, conflict_diagnostics = conflict_group_weight_lookup(
            manifest_path,
            conflicts_path,
            split_path,
            partition=("train" if training_scope == "fixed_split_validation" else None),
            minimum_size=max(record.sample_index for record in train_records) + 1,
        )
        conflict_weight_lookup = conflict_weight_lookup.to(device=device)
    visual_trust_lookup: torch.Tensor | None = None
    visual_trust_diagnostics: dict[str, float | int] | None = None
    if visual_trust_raw is not None and bool(visual_trust_raw.get("enabled", True)):
        cache_path = _resolve(base, str(visual_trust_raw["feature_cache"]))
        cache = load_feature_cache(cache_path)
        if cache.metadata.source != "train":
            raise ValueError("visual prototype weighting requires a training feature cache")
        if cache.metadata.manifest_sha256 != _sha256(manifest_path):
            raise ValueError("visual prototype cache manifest hash mismatch")
        if cache.metadata.split_sha256 != _sha256(split_path):
            raise ValueError("visual prototype cache split hash mismatch")
        _, expected_visual_trust_weight_sha256 = _model_settings(model_path)
        if cache.metadata.clip_weight_sha256 != expected_visual_trust_weight_sha256:
            raise ValueError("visual prototype cache CLIP weight hash mismatch")
        if cache.metadata.normalized is not True:
            raise ValueError("visual prototype weighting requires normalized CLIP features")
        cache_positions = {sample_id: index for index, sample_id in enumerate(cache.sample_ids)}
        if len(cache_positions) != len(cache.sample_ids):
            raise ValueError("visual prototype cache contains duplicate sample IDs")
        try:
            positions = torch.tensor([cache_positions[record.sample_id] for record in train_records])
        except KeyError as error:
            raise ValueError("training sample is missing from visual prototype cache") from error
        cached_labels = cache.labels.index_select(0, positions).long()
        record_labels = torch.tensor(
            [int(record.internal_class_index) for record in train_records], dtype=torch.long
        )
        if not torch.equal(cached_labels, record_labels):
            raise ValueError("visual prototype cache labels differ from the training manifest")
        visual_weights, visual_trust_diagnostics = visual_prototype_soft_weights(
            cache.features.index_select(0, positions),
            cached_labels,
            low_trust_fraction=float(visual_trust_raw.get("low_trust_fraction", 0.1)),
            min_weight=float(visual_trust_raw.get("min_weight", 0.5)),
            target_similarity_mix=float(visual_trust_raw.get("target_similarity_mix", 0.5)),
            chunk_size=int(visual_trust_raw.get("chunk_size", 4096)),
        )
        visual_trust_lookup = torch.full(
            (max(record.sample_index for record in train_records) + 1,),
            torch.nan,
            dtype=torch.float32,
        )
        visual_trust_lookup[
            torch.tensor([record.sample_index for record in train_records], dtype=torch.long)
        ] = visual_weights
    mapping = ClassMapping.from_csv(dataset_config.class_mapping)
    num_classes = len(mapping)
    elr_loss: ELRLoss | None = None
    if base_loss_kind == "elr":
        elr_state = ELRState(
            num_samples=max(record.sample_index for record in train_records) + 1,
            num_classes=num_classes,
            dtype=torch.float32,
            device="cpu",
            max_bytes=512 * 1024 * 1024,
        )
        elr_loss = ELRLoss(elr_state, beta=elr_beta, lambda_=elr_lambda, eps=elr_eps)
    training_class_counts = torch.bincount(
        torch.tensor(
            [int(record.internal_class_index) for record in train_records],
            dtype=torch.long,
        ),
        minlength=num_classes,
    )
    if bool((training_class_counts == 0).any()):
        raise ValueError("fixed training partition leaves one or more classes empty")
    sampling_raw = training.get("sampling", {"mode": "natural"})
    sampling_mode = str(sampling_raw.get("mode", "natural"))
    if sampling_mode not in {"natural", "class_balanced"}:
        raise ValueError(f"unsupported training sampling mode {sampling_mode!r}")
    sampling_weights: torch.Tensor | None = None
    sampling_diagnostics: dict[str, float | int | str] = {"mode": sampling_mode}
    if sampling_mode == "class_balanced":
        if confidence is not None or visual_trust_lookup is not None or label_smoothing > 0.0:
            raise ValueError("class-balanced sampling is a separate ablation")
        sampling_weights, balance_diagnostics = mild_class_balanced_sample_weights(
            torch.tensor([int(record.internal_class_index) for record in train_records]),
            num_classes=num_classes,
            power=float(sampling_raw.get("class_balance_power", 0.5)),
        )
        sampling_diagnostics.update(balance_diagnostics)
    logit_raw = raw.get("logit_adjustment", {"enabled": False})
    logit_adjustment: torch.Tensor | None = None
    logit_diagnostics: dict[str, float | int | str] | None = None
    if bool(logit_raw.get("enabled", False)):
        if (
            confidence is not None
            or visual_trust_lookup is not None
            or label_smoothing > 0.0
            or sampling_mode != "natural"
        ):
            raise ValueError("logit adjustment is a separate ablation")
        logit_adjustment, logit_diagnostics = train_prior_logit_adjustment(
            training_class_counts,
            tau=float(logit_raw.get("tau", 0.5)),
        )
    long_tail_stratification = stratify_classes(
        training_class_counts,
        head_fraction=0.2,
        mid_fraction=0.3,
        seed=int(dataset_config.split.seed),
    )
    validate_experiment_runtime_contract(
        raw,
        dataset_config,
        train_count=len(records),
        num_classes=num_classes,
    )
    weight_path, expected_weight_sha256 = _model_settings(model_path)
    clip_model, _, clip_metadata = load_clip_vit_b32(weight_path, expected_weight_sha256, device)
    provenance = (
        None
        if competition_stage is None
        else CheckpointProvenance(
            competition_stage=competition_stage,
            class_mapping_sha256=mapping.digest(),
            train_manifest_sha256=_sha256(manifest_path),
            split_sha256=_sha256(split_path),
            clip_weight_sha256=clip_metadata.weight_sha256,
            num_classes=num_classes,
        )
    )
    visual = clip_model.visual.float()
    del clip_model
    visual_input_resolution = int(raw.get("visual_input_resolution", 224))
    if visual_input_resolution != dataset_config.transforms.image_size:
        raise ValueError("visual input resolution must match the dataset transform contract")
    if visual_input_resolution in (320, 336):
        if raw.get("position_interpolation") != "bicubic_7x7_to_10x10":
            raise ValueError("320/336 visual input requires the frozen bicubic position interpolation")
        enable_interpolated_visual_resolution(visual, target_resolution=visual_input_resolution)
    elif visual_input_resolution != 224:
        raise ValueError("only official 224 or preregistered 320/336 visual input is supported")
    peft = raw["peft"]
    summary = configure_visual_peft(
        visual,
        str(peft["mode"]),
        lora_rank=int(peft.get("lora_rank", 4)),
        lora_alpha=float(peft.get("lora_alpha", 8.0)),
        lora_blocks=int(peft.get("lora_blocks", 1)),
    )
    head, head_checkpoint = load_classifier_head_checkpoint(
        head_checkpoint_path,
        feature_dim=clip_metadata.feature_dim,
        num_classes=num_classes,
        expected_clip_weight_sha256=clip_metadata.weight_sha256,
        expected_class_mapping_sha256=mapping.digest(),
        expected_split_sha256=_sha256(split_path),
        expected_provenance=provenance,
        forbid_full_train_initializer=training_scope == "fixed_split_validation",
    )
    head_config = head_checkpoint["head"]
    normalized_features = head_checkpoint["feature_cache_metadata"].get("normalized")
    if not isinstance(normalized_features, bool):
        raise ValueError("head checkpoint must declare whether cached features were normalized")
    head_checkpoint_sha256 = _sha256(head_checkpoint_path)
    model = PEFTImageClassifier(
        visual,
        head,
        normalize_features=normalized_features,
    ).to(device)
    visual_learning_rate = float(training["visual_learning_rate"])
    head_learning_rate = float(training["head_learning_rate"])
    if visual_learning_rate < 0 or head_learning_rate < 0:
        raise ValueError("learning rates must be non-negative")
    if head_learning_rate == 0:
        for parameter in head.parameters():
            parameter.requires_grad_(False)
    visual_parameters = [parameter for parameter in visual.parameters() if parameter.requires_grad]
    head_parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    parameter_groups = []
    if visual_parameters and visual_learning_rate > 0:
        parameter_groups.append({"params": visual_parameters, "lr": visual_learning_rate})
    if head_parameters and head_learning_rate > 0:
        parameter_groups.append({"params": head_parameters, "lr": head_learning_rate})
    if not parameter_groups:
        raise ValueError("training configuration enables no parameters with a positive learning rate")
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=float(training["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    trainable_ema = TrainableEMA(model, decay=ema_decay) if ema_enabled else None
    conflict_ids = (
        set()
        if training_scope == "full_train_fixed_epochs"
        else _conflict_ids(manifest_path, conflicts_path)
    )
    config_sha256 = _sha256(config_path)
    epoch = 0
    history: list[dict[str, Any]] = []
    best_top1 = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    raw_best_top1 = float("-inf")
    raw_best_state: dict[str, torch.Tensor] | None = None
    initial: dict[str, Any] | None = None

    if args.resume is not None:
        state = torch.load(args.resume.resolve(), map_location="cpu", weights_only=bool((raw.get("runtime") or {}).get("safe_numpy_rng_checkpoint", False)))
        if provenance is not None:
            validate_checkpoint_provenance(
                state,
                provenance,
                context="PEFT resume checkpoint",
            )
        if state["config_sha256"] != config_sha256:
            raise ValueError("PEFT resume configuration hash mismatch")
        load_trainable_state_dict(model, state["trainable_state"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        epoch = int(state["epoch"])
        history = state["history"]
        best_top1 = float(state["best_top1"])
        best_state = state["best_state"]
        if trainable_ema is not None:
            if "trainable_ema" not in state:
                raise ValueError("EMA resume checkpoint is missing EMA state")
            trainable_ema.load_state_dict(state["trainable_ema"])
            raw_best_top1 = float(state["raw_best_top1"])
            raw_best_state = state["raw_best_state"]
        elif "trainable_ema" in state:
            raise ValueError("non-EMA run cannot resume an EMA checkpoint")
        initial = state.get("initial_validation")
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.random.set_rng_state(state["torch_rng_state"])
        if "cuda_rng_state_all" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        if confidence is not None:
            if "confidence_weighting" not in state:
                raise ValueError("M1 resume checkpoint is missing confidence state")
            confidence.load_state_dict(state["confidence_weighting"])
        elif "confidence_weighting" in state:
            raise ValueError("non-M1 run cannot resume an M1 confidence checkpoint")
        if elr_loss is not None:
            if "elr_state" not in state:
                raise ValueError("ELR resume checkpoint is missing ELR state")
            elr_loss.state.load_state_dict(state["elr_state"])
        elif "elr_state" in state:
            raise ValueError("non-ELR run cannot resume an ELR checkpoint")

    if epoch == 0 and training_scope == "fixed_split_validation":
        assert validation_dataset is not None
        initial = _evaluate(
            model,
            validation_records,
            validation_dataset,
            loader=validation_loader,
            batch_size=batch_size,
            workers=workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            device=device,
            num_classes=num_classes,
            conflict_ids=conflict_ids,
            stratification=long_tail_stratification,
        )
        expected = float(raw["expected_initial_top1"])
        tolerance = float(raw.get("initial_top1_tolerance", 0.002))
        if abs(float(initial["all"]["top1"]) - expected) > tolerance:
            raise RuntimeError(
                f"raw-image head parity failed: got {initial['all']['top1']}, expected {expected}±{tolerance}"
            )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_started = time.perf_counter()
    early_stop: dict[str, Any] | None = None
    while epoch < epochs:
        train_dataset.set_epoch(epoch)
        model.train()
        confidence_active = confidence is not None and epoch >= confidence_warmup_epochs
        applied_confidence = confidence.diagnostics() if confidence is not None else None
        loader = _loader(
            train_dataset,
            batch_size=batch_size,
            workers=workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            shuffle=True,
            seed=seed + epoch,
            device=device,
            sample_weights=sampling_weights,
        )
        train_started = time.perf_counter()
        loss_sum = (
            torch.zeros((), dtype=torch.float64, device=device)
            if defer_epoch_metrics
            else 0.0
        )
        count = 0
        record_gradient_diagnostics = bool((raw.get("runtime") or {}).get("record_gradient_diagnostics", False))
        gradient_norms: list[torch.Tensor] = []
        diagnostic_initial_scale = scaler.get_scale() if record_gradient_diagnostics else None
        epoch_sample_indices: list[torch.Tensor] = []
        epoch_sample_losses: list[torch.Tensor] = []
        epoch_observed_probabilities: list[torch.Tensor] = []
        epoch_predictions: list[torch.Tensor] = []
        epoch_sampling_counts = torch.zeros(
            num_classes,
            dtype=torch.long,
            device=device if defer_epoch_metrics else torch.device("cpu"),
        )
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            conflict_batch_weights = None
            if conflict_weight_lookup is not None:
                conflict_batch_weights = conflict_weight_lookup.index_select(
                    0, batch["sample_index"].long().to(device, non_blocking=True)
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
                loss_logits = (
                    logits
                    if logit_adjustment is None
                    else logits + logit_adjustment.to(device=device, dtype=logits.dtype)
                )
                per_sample_loss = _base_loss_per_sample(
                    loss_logits,
                    labels,
                    kind=base_loss_kind,
                    q=base_loss_q,
                    label_smoothing=label_smoothing,
                    sce_alpha=sce_alpha,
                    sce_beta=sce_beta,
                    sce_log_zero=sce_log_zero,
                    nce_alpha=nce_alpha,
                    nce_beta=nce_beta,
                    nce_log_zero=nce_log_zero,
                    elr_loss=elr_loss,
                    sample_indices=batch["sample_index"],
                )
                batch_weights: torch.Tensor | None = None
                if confidence_active:
                    batch_weights = confidence.weights_for(batch["sample_index"]).to(
                        device, non_blocking=True
                    )
                    if conflict_batch_weights is not None:
                        batch_weights = batch_weights * conflict_batch_weights
                elif visual_trust_lookup is not None:
                    batch_weights = visual_trust_lookup.index_select(
                        0, batch["sample_index"].long()
                    ).to(device, non_blocking=True)
                    if conflict_batch_weights is not None:
                        batch_weights = batch_weights * conflict_batch_weights
                elif conflict_batch_weights is not None:
                    batch_weights = conflict_batch_weights
                loss = _weighted_base_loss(per_sample_loss, batch_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("PEFT training produced a non-finite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = nn.utils.clip_grad_norm_(
                [*visual_parameters, *head_parameters],
                float(training["gradient_clip_norm"]),
            )
            if record_gradient_diagnostics:
                gradient_norms.append(gradient_norm.detach())
            previous_scale = scaler.get_scale() if trainable_ema is not None else None
            scaler.step(optimizer)
            scaler.update()
            if trainable_ema is not None and optimizer_step_completed(
                scaler_enabled=scaler.is_enabled(),
                before=previous_scale,
                after=scaler.get_scale(),
            ):
                trainable_ema.update()
            if defer_epoch_metrics:
                loss_sum += loss.detach().to(dtype=torch.float64) * len(labels)
            else:
                loss_sum += float(loss.detach()) * len(labels)
            count += len(labels)
            if defer_epoch_metrics:
                epoch_sampling_counts += torch.bincount(labels, minlength=num_classes)
            else:
                epoch_sampling_counts += torch.bincount(
                    labels.detach().cpu(), minlength=num_classes
                )
            if confidence is not None:
                detached_logits = logits.detach().float()
                probabilities = detached_logits.softmax(dim=1)
                epoch_sample_indices.append(batch["sample_index"].detach().long().cpu())
                sample_losses = F.cross_entropy(detached_logits, labels, reduction="none")
                observed_probabilities = probabilities.gather(
                    1, labels[:, None]
                ).squeeze(1)
                predictions = detached_logits.argmax(dim=1)
                if defer_epoch_metrics:
                    epoch_sample_losses.append(sample_losses)
                    epoch_observed_probabilities.append(observed_probabilities)
                    epoch_predictions.append(predictions)
                else:
                    epoch_sample_losses.append(sample_losses.cpu())
                    epoch_observed_probabilities.append(observed_probabilities.cpu())
                    epoch_predictions.append(predictions.cpu())
        train_seconds = time.perf_counter() - train_started
        if defer_epoch_metrics:
            loss_sum_value = float(loss_sum.item())
            epoch_sampling_counts = epoch_sampling_counts.cpu()
        else:
            loss_sum_value = float(loss_sum)
        next_confidence = None
        next_effective_quality = None
        if confidence is not None:
            completed_indices = torch.cat(epoch_sample_indices)
            completed_losses = torch.cat(epoch_sample_losses)
            completed_observed_probabilities = torch.cat(epoch_observed_probabilities)
            completed_predictions = torch.cat(epoch_predictions)
            if defer_epoch_metrics:
                completed_losses = completed_losses.cpu()
                completed_observed_probabilities = completed_observed_probabilities.cpu()
                completed_predictions = completed_predictions.cpu()
            next_confidence = confidence.update_epoch(
                completed_indices,
                completed_losses,
                completed_observed_probabilities,
                completed_predictions,
            )
            next_effective_quality = confidence.effective_quality_profile(
                completed_indices,
                completed_losses,
            )
        scheduler.step()
        epoch += 1
        row = {
            "epoch": epoch,
            "train_loss": loss_sum_value / count,
            "train_seconds": train_seconds,
            "train_samples_per_second": count / train_seconds,
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            "sampling": {
                "mode": sampling_mode,
                "sample_count": int(epoch_sampling_counts.sum()),
                "class_count_min": int(epoch_sampling_counts.min()),
                "class_count_median": float(epoch_sampling_counts.float().median()),
                "class_count_max": int(epoch_sampling_counts.max()),
                "zero_sample_class_count": int((epoch_sampling_counts == 0).sum()),
            },
        }
        if training_scope == "fixed_split_validation":
            assert validation_dataset is not None
            if record_gradient_diagnostics:
                norms = torch.stack(gradient_norms).float()
                finite_norms = norms[torch.isfinite(norms)]
                row["gradient_diagnostics"] = {
                    "pre_clip_finite_norm_mean": float(finite_norms.mean().item()) if len(finite_norms) else None,
                    "pre_clip_finite_norm_max": float(finite_norms.max().item()) if len(finite_norms) else None,
                    "finite_norm_clip_fraction": float((finite_norms > float(training["gradient_clip_norm"])).float().mean().item()) if len(finite_norms) else None,
                    "all_finite": bool(torch.isfinite(norms).all().item()),
                    "nonfinite_norm_count": int((~torch.isfinite(norms)).sum().item()),
                    "initial_scaler_scale": diagnostic_initial_scale,
                    "final_scaler_scale": scaler.get_scale(),
                    "note": "raw nonfinite gradients may be normal AMP overflow; scaler skips such optimizer steps; inspect separately",
                    "batch_count": len(gradient_norms),
                    "diagnostic_only": True,
                }
            validation = _evaluate(
                model,
                validation_records,
                validation_dataset,
                loader=validation_loader,
                batch_size=batch_size,
                workers=workers,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
                device=device,
                num_classes=num_classes,
                conflict_ids=conflict_ids,
                stratification=long_tail_stratification,
            )
            row["validation"] = validation
            if trainable_ema is not None:
                with trainable_ema.applied():
                    row["ema_validation"] = _evaluate(
                        model,
                        validation_records,
                        validation_dataset,
                        loader=validation_loader,
                        batch_size=batch_size,
                        workers=workers,
                        prefetch_factor=prefetch_factor,
                        persistent_workers=persistent_workers,
                        device=device,
                        num_classes=num_classes,
                        conflict_ids=conflict_ids,
                        stratification=long_tail_stratification,
                    )
        if confidence is not None:
            row["confidence_weighting"] = {
                "active": confidence_active,
                "applied": applied_confidence,
                "next": next_confidence,
                "next_effective_quality": next_effective_quality,
            }
            if (
                bool(confidence_raw.get("stop_on_effective_quality_l2", False))
                and next_effective_quality is not None
                and next_effective_quality["decision"] == "L2"
            ):
                early_stop = {
                    "epoch": epoch,
                    "reason": "next-epoch confidence weights triggered L2 effective-quality gate",
                    "decision": "L2",
                }
                row["early_stop"] = early_stop
        history.append(row)
        if training_scope == "fixed_split_validation":
            if trainable_ema is not None:
                raw_top1 = float(row["validation"]["all"]["top1"])
                if raw_top1 > raw_best_top1:
                    raw_best_top1 = raw_top1
                    raw_best_state = copy.deepcopy(trainable_state_dict(model))
            selection_validation = row.get("ema_validation", row["validation"])
            top1 = float(selection_validation["all"]["top1"])
            if top1 > best_top1:
                best_top1 = top1
                best_state = (
                    trainable_ema.trainable_state_dict()
                    if trainable_ema is not None
                    else copy.deepcopy(trainable_state_dict(model))
                )
        checkpoint: dict[str, Any] = {
            "config_sha256": config_sha256,
            "visual_input_resolution": visual_input_resolution,
            "position_interpolation": raw.get("position_interpolation"),
            "epoch": epoch,
            "trainable_state": trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
            "initial_validation": initial,
            "best_top1": best_top1,
            "best_state": best_state,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.random.get_rng_state().clone(),
        }
        if provenance is not None:
            checkpoint.update(provenance.to_dict())
        if torch.cuda.is_available():
            checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        if confidence is not None:
            checkpoint["confidence_weighting"] = confidence.state_dict()
        if trainable_ema is not None:
            checkpoint["trainable_ema"] = trainable_ema.state_dict()
            checkpoint["raw_best_top1"] = raw_best_top1
            checkpoint["raw_best_state"] = raw_best_state
        if elr_loss is not None:
            checkpoint["elr_state"] = elr_loss.state.state_dict()
        if bool((raw.get("runtime") or {}).get("safe_numpy_rng_checkpoint", False)):
            # MT19937 list round-trips through np.random.set_state exactly while
            # allowing PyTorch weights_only CPU loading without NumPy globals.
            name, keys, position, has_gauss, cached = checkpoint["numpy_rng_state"]
            checkpoint["numpy_rng_state"] = (name, keys.tolist(), position, has_gauss, cached)
        _save_atomic(checkpoint, output / "last.pt")
        if training_scope == "fixed_split_validation":
            long_tail = row["validation"]["long_tail"]
            print(
                json.dumps(
                    {
                        "stage": "fixed_split_peft",
                        "status": "epoch_complete",
                        "epoch": epoch,
                        "epochs": epochs,
                        "train_loss": row["train_loss"],
                        "top1": row["validation"]["all"]["top1"],
                        "ema_top1": (
                            row["ema_validation"]["all"]["top1"]
                            if trainable_ema is not None
                            else None
                        ),
                        "macro_accuracy": long_tail["macro_accuracy"],
                        "tail_accuracy": long_tail["strata"]["tail"]["accuracy"],
                        "zero_recall_class_count": long_tail["zero_recall_class_count"],
                        "sampling_zero_class_count": row["sampling"][
                            "zero_sample_class_count"
                        ],
                        "effective_quality_decision": (
                            next_effective_quality["decision"]
                            if next_effective_quality is not None
                            else None
                        ),
                        "effective_quality_min_ratio": (
                            next_effective_quality["summary"][
                                "min_effective_to_raw_ratio"
                            ]
                            if next_effective_quality is not None
                            else None
                        ),
                        "effective_quality_p10_ratio": (
                            next_effective_quality["summary"][
                                "p10_effective_to_raw_ratio"
                            ]
                            if next_effective_quality is not None
                            else None
                        ),
                        "early_stop": early_stop,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if early_stop is not None:
            break
        if training_scope == "full_train_fixed_epochs":
            print(
                json.dumps(
                    {
                        "stage": "full_train_peft",
                        "status": "epoch_complete",
                        "epoch": epoch,
                        "epochs": epochs,
                        "train_loss": row["train_loss"],
                        "train_seconds": train_seconds,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if training_scope == "full_train_fixed_epochs":
        selected_state = (
            trainable_ema.trainable_state_dict()
            if trainable_ema is not None
            else trainable_state_dict(model)
        )
        selected_name = "final.pt"
        selected_top1 = None
        selection_rule = f"fixed final epoch {epochs}; no validation selection"
    else:
        if best_state is None:
            raise RuntimeError("PEFT training produced no best state")
        selected_state = best_state
        selected_name = "best.pt"
        selected_top1 = best_top1
        selection_rule = (
            "best EMA fixed-split validation Top-1"
            if trainable_ema is not None
            else "best fixed-split validation Top-1"
        )
    selected_checkpoint: dict[str, Any] = {
        "config_sha256": config_sha256,
        "visual_input_resolution": visual_input_resolution,
        "position_interpolation": raw.get("position_interpolation"),
        "mode": peft["mode"],
        "clip_weight_sha256": clip_metadata.weight_sha256,
        "class_mapping_sha256": mapping.digest(),
        "split_sha256": _sha256(split_path),
        "head_initializer_sha256": head_checkpoint_sha256,
        "head_kind": str(head_config["kind"]),
        "normalize_features": normalized_features,
        "trainable_state": selected_state,
        "best_top1": selected_top1,
        "training_scope": training_scope,
        "data_loader": {
            "workers": workers,
            "prefetch_factor": prefetch_factor if workers > 0 else None,
            "persistent_workers": persistent_workers if workers > 0 else False,
            "pin_memory": device.type == "cuda",
        },
        "runtime": runtime_settings,
        "selection_rule": selection_rule,
    }
    if provenance is not None:
        selected_checkpoint.update(provenance.to_dict())
    if trainable_ema is not None:
        selected_checkpoint["trainable_ema"] = {
            "enabled": True, "decay": trainable_ema.decay, "updates": trainable_ema.updates
        }
    _save_atomic(selected_checkpoint, output / selected_name)
    if trainable_ema is not None and training_scope == "fixed_split_validation":
        if raw_best_state is None:
            raise RuntimeError("EMA run did not produce a raw diagnostic checkpoint")
        raw_checkpoint = dict(selected_checkpoint)
        raw_checkpoint["trainable_state"] = raw_best_state
        raw_checkpoint["best_top1"] = raw_best_top1
        raw_checkpoint["trainable_ema"] = None
        raw_checkpoint["selection_rule"] = "raw best fixed-split validation Top-1; diagnostic only"
        _save_atomic(raw_checkpoint, output / "raw_best.pt")
    total_seconds = time.perf_counter() - total_started
    environment = capture_environment_identity(
        ROOT,
        torch_module=torch,
        device=device,
        clip_weight_sha256=clip_metadata.weight_sha256,
    )
    git_commit = _git_commit(ROOT)
    metrics = {
        "visual_input_resolution": visual_input_resolution,
        "position_interpolation": raw.get("position_interpolation"),
        "scope": (
            f"{competition_stage or 'stage1-legacy'} full official train PEFT final retrain; "
            "test data not accessed"
            if training_scope == "full_train_fixed_epochs"
            else f"{competition_stage or 'stage1-legacy'} train/validation PEFT short screening; "
            "test data not accessed"
        ),
        "run_id": raw["run_id"],
        "training_scope": training_scope,
        "runtime": runtime_settings,
        "data_loader": {
            "workers": workers,
            "prefetch_factor": prefetch_factor if workers > 0 else None,
            "persistent_workers": persistent_workers if workers > 0 else False,
            "pin_memory": device.type == "cuda",
        },
        "selection_rule": selection_rule,
        "peft": peft,
        "confidence_weighting": confidence_raw,
        "conflict_loss": conflict_raw,
        "conflict_diagnostics": conflict_diagnostics,
        "early_stop": early_stop,
        "visual_prototype_weighting": visual_trust_raw,
        "visual_prototype_diagnostics": visual_trust_diagnostics,
        "base_loss": base_loss_raw,
        "label_smoothing": label_smoothing,
        "sampling": sampling_diagnostics,
        "logit_adjustment": logit_diagnostics,
        "initial_validation": initial,
        "history": history,
        "best_top1": selected_top1,
        "duration_seconds": total_seconds,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0,
        "train_count": len(train_records),
        "validation_count": len(validation_records),
        "total_visual_parameters": summary.total_visual_parameters,
        "trainable_visual_parameters": summary.trainable_visual_parameters,
        "trainable_total_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "selected_checkpoint": selected_name,
        "selected_checkpoint_sha256": _sha256(output / selected_name),
        "last_checkpoint_sha256": _sha256(output / "last.pt"),
        "config_sha256": config_sha256,
        "head_initializer": {
            "path": str(head_checkpoint_path),
            "sha256": head_checkpoint_sha256,
            "kind": str(head_config["kind"]),
            "normalize_features": normalized_features,
        },
        "dataset_config_sha256": _sha256(dataset_path),
        "manifest_sha256": _sha256(manifest_path),
        "split_sha256": _sha256(split_path),
        "git_commit": git_commit,
        "git_commit_source": (
            "verified_git_checkout_or_launcher_receipt"
            if git_commit != "unknown_archive"
            else "unknown_archive; use per-file source hashes in the queue receipt"
        ),
        "environment": environment,
        "environment_hash": environment_hash(environment),
    }
    if provenance is not None:
        metrics["checkpoint_provenance"] = provenance.to_dict()
    if selected_name == "best.pt":
        metrics["best_checkpoint_sha256"] = metrics["selected_checkpoint_sha256"]
    else:
        metrics["final_checkpoint_sha256"] = metrics["selected_checkpoint_sha256"]
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copy2(config_path, output / "config.yaml")
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
