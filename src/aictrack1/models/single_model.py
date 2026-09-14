"""Rebuild one frozen single-model checkpoint for migration stress diagnostics.

The helper mirrors the final inference rules so a diagnostic forward pass uses
exactly the same composition as the submission path: official OpenAI CLIP
ViT-B/32 weights, one classifier head, one LoRA configuration, and LoRA folded
before the forward pass.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from ..engine.peft import load_trainable_state_dict
from .clip_vit_b32 import load_clip_vit_b32
from .heads import load_classifier_head_checkpoint
from .peft import PEFTImageClassifier, configure_visual_peft, fold_visual_lora

SUPPORTED_PEFT_MODES = frozenset({"lora_last_attention", "lora_last_n_attention"})


@dataclass(frozen=True)
class DiagnosticModel:
    """A fitted model with the provenance needed for an auditable report."""

    model: PEFTImageClassifier
    device: torch.device
    class_mapping_digest: str
    clip_weight_sha256: str
    feature_dim: int
    num_classes: int
    normalized_features: bool
    run_id: str
    config_sha256: str
    checkpoint_sha256: str
    git_commit: str

    def provenance(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "official_clip_weight_sha256": self.clip_weight_sha256,
            "class_mapping_sha256": self.class_mapping_digest,
            "git_commit": self.git_commit,
            "device": str(self.device),
            "lora_folded_for_single_model_inference": True,
        }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_relative(base: Path, value: str | Path) -> Path:
    """Resolve a relative setting against a base directory.

    ``base`` must be a directory; passing a file path silently collapses the
    ``..`` components in the joined path, so the mistake is rejected loudly.
    """
    base = Path(base)
    if base.suffix:
        raise ValueError(f"resolve base must be a directory, got {base}")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def model_settings(path: Path) -> tuple[Path, str]:
    """Read the frozen model contract: OpenAI ViT-B/32 with downloads disabled."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = document.get("model", document)
    if raw.get("provider") != "OpenAI" or raw.get("name") != "ViT-B/32":
        raise ValueError("model config must declare OpenAI ViT-B/32")
    if raw.get("runtime_downloads_allowed") is not False:
        raise ValueError("runtime downloads must be disabled")
    return resolve_relative(path.parent, str(raw["weight_path"])), str(
        raw["expected_weight_sha256"]
    )


def read_partition_assignments(path: Path, column: str) -> dict[str, str]:
    """Read a `sample_id` plus partition-column CSV, rejecting duplicated IDs."""
    assignments: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"sample_id", column}.issubset(reader.fieldnames):
            raise ValueError(f"partition file requires sample_id and {column} columns")
        for row in reader:
            sample_id = str(row["sample_id"])
            if sample_id in assignments:
                raise ValueError(f"duplicate sample_id in partition file: {sample_id}")
            assignments[sample_id] = str(row[column])
    return assignments


def load_diagnostic_model(
    *,
    config_path: Path,
    checkpoint_path: Path,
    class_mapping_digest: str,
    num_classes: int,
    split_sha256: str,
    device: torch.device,
    git_commit: str,
) -> DiagnosticModel:
    """Rebuild the fitted model from a frozen checkpoint, verifying every hash."""
    config_path = config_path.resolve()
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = document.get("experiment", document)
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_sha256 = sha256_file(config_path)
    if checkpoint.get("config_sha256") not in _text_sha256_variants(config_path):
        raise ValueError("checkpoint and experiment configuration hashes differ")
    peft = experiment["peft"]
    if checkpoint.get("mode") != peft.get("mode"):
        raise ValueError("checkpoint and experiment PEFT modes differ")
    if peft.get("mode") not in SUPPORTED_PEFT_MODES:
        raise ValueError("diagnostics require the selected LoRA PEFT mode")
    if checkpoint.get("class_mapping_sha256") != class_mapping_digest:
        raise ValueError("checkpoint and class mapping hashes differ")
    if int(num_classes) <= 0:
        raise ValueError("num_classes must be positive")

    model_path = resolve_relative(config_path.parent, str(experiment["model_config"]))
    weight_path, expected_weight_sha256 = model_settings(model_path)
    clip_model, _, clip_metadata = load_clip_vit_b32(weight_path, expected_weight_sha256, device)
    if checkpoint.get("clip_weight_sha256") != clip_metadata.weight_sha256:
        raise ValueError("checkpoint and official CLIP weight hashes differ")
    visual = clip_model.visual.float()
    del clip_model
    configure_visual_peft(
        visual,
        str(peft["mode"]),
        lora_rank=int(peft.get("lora_rank", 4)),
        lora_alpha=float(peft.get("lora_alpha", 8.0)),
        lora_blocks=int(peft.get("lora_blocks", 1)),
    )
    num_classes = int(num_classes)
    configured = experiment.get("num_classes", checkpoint.get("num_classes"))
    if configured is not None and int(configured) != num_classes:
        raise ValueError("experiment or checkpoint num_classes disagrees with the class mapping")
    head_setting = experiment.get("head_checkpoint", experiment.get("a1_head_checkpoint"))
    if head_setting is None:
        raise ValueError("experiment must define head_checkpoint or legacy a1_head_checkpoint")
    head_path = resolve_relative(config_path.parent, str(head_setting))
    head, head_checkpoint = load_classifier_head_checkpoint(
        head_path,
        feature_dim=clip_metadata.feature_dim,
        num_classes=num_classes,
        expected_clip_weight_sha256=clip_metadata.weight_sha256,
        expected_class_mapping_sha256=class_mapping_digest,
        expected_split_sha256=split_sha256,
    )
    normalized = head_checkpoint["feature_cache_metadata"].get("normalized")
    if not isinstance(normalized, bool):
        raise ValueError("head checkpoint must declare whether cached features were normalized")
    checkpoint_normalized = checkpoint.get("normalize_features")
    if checkpoint_normalized is not None and checkpoint_normalized != normalized:
        raise ValueError("checkpoint and classifier head feature normalization differ")
    expected_head_sha256 = checkpoint.get("head_initializer_sha256")
    if expected_head_sha256 is not None and expected_head_sha256 != sha256_file(head_path):
        raise ValueError("checkpoint and classifier head initializer hashes differ")
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    model = PEFTImageClassifier(visual, head, normalize_features=normalized).to(device)
    load_trainable_state_dict(model, checkpoint["trainable_state"])
    fold_visual_lora(model.visual)
    model.eval()
    return DiagnosticModel(
        model=model,
        device=device,
        class_mapping_digest=class_mapping_digest,
        clip_weight_sha256=clip_metadata.weight_sha256,
        feature_dim=clip_metadata.feature_dim,
        num_classes=num_classes,
        normalized_features=normalized,
        run_id=str(experiment["run_id"]),
        config_sha256=config_sha256,
        checkpoint_sha256=sha256_file(checkpoint_path),
        git_commit=git_commit,
    )


def _text_sha256_variants(path: Path) -> set[str]:
    raw = path.read_bytes()
    normalized = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return {
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        hashlib.sha256(normalized.replace("\n", "\r\n").encode("utf-8")).hexdigest(),
    }


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
