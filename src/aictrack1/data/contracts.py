"""Explicit data contracts that never infer competition semantics implicitly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ALLOWED_STAGES = {"synthetic", "preliminary", "rematch", "semifinal"}
ALLOWED_LABEL_SOURCES = {"manifest", "directory", "none"}


@dataclass(frozen=True)
class SourceConfig:
    name: str
    root: Path
    label_source: str
    manifest: Path | None = None
    class_from_parent: bool = False


@dataclass(frozen=True)
class AuditConfig:
    output_root: Path
    workers: int = 4
    near_duplicate_hamming: int = 3
    near_duplicate_max_bucket: int = 500


@dataclass(frozen=True)
class SplitConfig:
    validation_fraction: float = 0.2
    seed: int = 42
    min_train_per_class: int = 1


@dataclass(frozen=True)
class DatasetConfig:
    config_path: Path
    stage: str
    dataset_root: Path
    class_mapping: Path
    extensions: tuple[str, ...]
    allow_truncated_images: bool
    sources: dict[str, SourceConfig]
    audit: AuditConfig
    split: SplitConfig


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{context}: missing required field {key!r}")
    return mapping[key]


def _parse_source(name: str, raw: dict[str, Any], dataset_root: Path) -> SourceConfig:
    label_source = str(_require(raw, "label_source", f"source {name}"))
    if label_source not in ALLOWED_LABEL_SOURCES:
        raise ValueError(f"source {name}: unsupported label_source {label_source!r}")
    root = _resolve(dataset_root, str(_require(raw, "root", f"source {name}")))
    manifest_value = raw.get("manifest")
    manifest = _resolve(dataset_root, str(manifest_value)) if manifest_value else None
    class_from_parent = bool(raw.get("class_from_parent", False))
    if label_source == "manifest" and manifest is None:
        raise ValueError(f"source {name}: manifest label_source requires manifest")
    if label_source == "directory" and not class_from_parent:
        raise ValueError(f"source {name}: directory labels require class_from_parent: true")
    if label_source == "none" and (manifest is not None or class_from_parent):
        raise ValueError(f"source {name}: unlabeled source cannot define label metadata")
    if name == "test" and label_source != "none":
        raise ValueError("test source must use label_source: none")
    return SourceConfig(name, root, label_source, manifest, class_from_parent)


def load_dataset_config(path: str | Path) -> DatasetConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a mapping")
    raw = raw.get("dataset", raw)
    if not isinstance(raw, dict):
        raise ValueError("dataset configuration must be a mapping")

    stage = str(_require(raw, "stage", "dataset"))
    if stage not in ALLOWED_STAGES:
        raise ValueError(f"unsupported stage {stage!r}; expected one of {sorted(ALLOWED_STAGES)}")
    base = config_path.parent
    dataset_root = _resolve(base, str(_require(raw, "dataset_root", "dataset")))
    class_mapping = _resolve(dataset_root, str(_require(raw, "class_mapping", "dataset")))
    extensions_raw = _require(raw, "extensions", "dataset")
    if not isinstance(extensions_raw, list) or not extensions_raw:
        raise ValueError("dataset.extensions must be a non-empty list")
    extensions = tuple(sorted({str(ext).lower() if str(ext).startswith(".") else f".{str(ext).lower()}" for ext in extensions_raw}))

    sources_raw = _require(raw, "sources", "dataset")
    if not isinstance(sources_raw, dict) or "train" not in sources_raw:
        raise ValueError("dataset.sources must contain an explicit train source")
    sources = {name: _parse_source(name, value, dataset_root) for name, value in sources_raw.items()}

    audit_raw = _require(raw, "audit", "dataset")
    audit = AuditConfig(
        output_root=_resolve(base, str(_require(audit_raw, "output_root", "audit"))),
        workers=int(audit_raw.get("workers", 4)),
        near_duplicate_hamming=int(audit_raw.get("near_duplicate_hamming", 3)),
        near_duplicate_max_bucket=int(audit_raw.get("near_duplicate_max_bucket", 500)),
    )
    if not 0 <= audit.near_duplicate_hamming <= 3:
        raise ValueError("near_duplicate_hamming must be between 0 and 3 for the LSH guarantee")
    if audit.workers < 1:
        raise ValueError("audit.workers must be at least 1")
    if audit.near_duplicate_max_bucket < 2:
        raise ValueError("near_duplicate_max_bucket must be at least 2")

    split_raw = _require(raw, "split", "dataset")
    split = SplitConfig(
        validation_fraction=float(split_raw.get("validation_fraction", 0.2)),
        seed=int(split_raw.get("seed", 42)),
        min_train_per_class=int(split_raw.get("min_train_per_class", 1)),
    )
    if not 0 < split.validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if split.min_train_per_class < 1:
        raise ValueError("min_train_per_class must be at least 1")

    source_roots = [source.root for source in sources.values()]
    if audit.output_root == dataset_root or audit.output_root.is_relative_to(dataset_root):
        raise ValueError("audit output_root cannot be inside dataset_root")
    for source_root in source_roots:
        if audit.output_root == source_root or audit.output_root.is_relative_to(source_root):
            raise ValueError("audit output_root cannot be inside a source root")

    return DatasetConfig(
        config_path=config_path,
        stage=stage,
        dataset_root=dataset_root,
        class_mapping=class_mapping,
        extensions=extensions,
        allow_truncated_images=bool(raw.get("allow_truncated_images", False)),
        sources=sources,
        audit=audit,
        split=split,
    )
