"""Explicit data contracts that never infer competition semantics implicitly."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ALLOWED_STAGES = {"synthetic", "preliminary", "rematch", "semifinal"}
ALLOWED_LABEL_SOURCES = {"manifest", "directory", "none"}
ALLOWED_CLASS_MAPPING_PROVENANCE = {"official_file", "train_directory_names"}
ALLOWED_COMPETITION_STAGES = {"stage1", "stage2", "stage3"}
ALLOWED_SUBMISSION_FILENAME_MODES = {"basename", "relative_path"}
STAGE_REFERENCE_FIELDS = {
    "dataset_config",
    "dataset_root",
    "manifest",
    "split",
    "label_conflicts",
    "class_mapping",
    "feature_cache",
    "head_checkpoint",
    "a1_head_checkpoint",
    "checkpoint",
    "output_dir",
    "output_root",
    "weight_path",
}


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
    fail_on_bad_files: bool = True
    fail_on_label_conflicts: bool = True


@dataclass(frozen=True)
class SplitConfig:
    validation_fraction: float = 0.2
    seed: int = 42
    min_train_per_class: int = 1


@dataclass(frozen=True)
class TransformConfig:
    image_size: int = 224
    resize_size: int = 224
    train_crop_scale_min: float = 0.8
    train_horizontal_flip_probability: float = 0.5


@dataclass(frozen=True)
class SubmissionContract:
    filename_mode: str = "basename"
    official_class_id_pattern: str | None = None


@dataclass(frozen=True)
class DatasetConfig:
    config_path: Path
    competition_stage: str | None
    contract_confirmed: bool
    expected_train_count: int | None
    expected_test_count: int | None
    expected_num_classes: int | None
    stage: str
    dataset_root: Path
    class_mapping: Path
    class_mapping_provenance: str
    extensions: tuple[str, ...]
    allow_truncated_images: bool
    sources: dict[str, SourceConfig]
    audit: AuditConfig
    split: SplitConfig
    transforms: TransformConfig
    submission: SubmissionContract


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{context}: missing required field {key!r}")
    return mapping[key]


def _optional_bool(mapping: dict[str, Any], key: str, default: bool, context: str) -> bool:
    value = mapping.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be a boolean")
    return value


def _optional_positive_int(
    mapping: dict[str, Any], key: str, context: str
) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context}.{key} must be a positive integer or null")
    if value < 1:
        raise ValueError(f"{context}.{key} must be a positive integer or null")
    return value


def validate_experiment_runtime_contract(
    experiment: Mapping[str, Any],
    dataset: DatasetConfig,
    *,
    train_count: int | None = None,
    test_count: int | None = None,
    num_classes: int | None = None,
) -> str | None:
    """Validate the executable stage/count contract for new stage-scoped runs.

    Historical Stage 1 configurations predate ``competition_stage`` and remain
    loadable. Every new stage-scoped configuration must declare the stage in both
    the experiment and dataset contracts, confirm the audited dataset contract,
    and freeze all three official counts before execution.
    """

    raw_stage = experiment.get("competition_stage")
    if raw_stage is None:
        if dataset.competition_stage is not None:
            raise ValueError(
                "experiment must declare competition_stage when the dataset does"
            )
        return None
    stage = str(raw_stage).lower()
    if stage not in ALLOWED_COMPETITION_STAGES:
        raise ValueError(
            f"experiment: unsupported competition_stage {raw_stage!r}; "
            f"expected one of {sorted(ALLOWED_COMPETITION_STAGES)}"
        )
    if dataset.competition_stage != stage:
        raise ValueError(
            "experiment and dataset competition_stage mismatch: "
            f"{stage!r} != {dataset.competition_stage!r}"
        )
    validate_competition_stage_references(stage, experiment, context="experiment")
    if experiment.get("execution_enabled", True) is not True:
        return stage
    if not dataset.contract_confirmed:
        raise ValueError("dataset contract_confirmed must be true before execution")

    expected = {
        "train_count": dataset.expected_train_count,
        "test_count": dataset.expected_test_count,
        "num_classes": dataset.expected_num_classes,
    }
    missing = [name for name, value in expected.items() if value is None]
    if missing:
        raise ValueError(
            "confirmed dataset contract is missing required expectations: "
            + ", ".join(missing)
        )
    actual = {
        "train_count": train_count,
        "test_count": test_count,
        "num_classes": num_classes,
    }
    for name, value in actual.items():
        if value is not None and int(value) != expected[name]:
            raise ValueError(
                f"dataset {name} mismatch: actual {int(value)} != expected {expected[name]}"
            )
    return stage


def validate_competition_stage_references(
    competition_stage: str,
    document: Mapping[str, Any],
    *,
    context: str = "configuration",
) -> None:
    """Reject path-like references to assets from another competition stage.

    This is deliberately a static guard: it examines declared references without
    requiring the referenced data, manifests, or checkpoints to exist. Stage 1
    configurations predate ``competition_stage`` and remain valid; every new Stage 2
    template/config must opt in and is then forbidden from naming Stage 1 assets.
    """

    stage = str(competition_stage).lower()
    if stage not in ALLOWED_COMPETITION_STAGES:
        raise ValueError(
            f"{context}: unsupported competition_stage {competition_stage!r}; "
            f"expected one of {sorted(ALLOWED_COMPETITION_STAGES)}"
        )

    forbidden_markers = {
        other
        for other in ALLOWED_COMPETITION_STAGES
        if other != stage
    }
    violations: list[str] = []

    def mentions_stage(value: str, marker: str) -> bool:
        normalized = value.replace("\\", "/").lower()
        stage_token = re.compile(rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])")
        dataset_token = re.compile(
            rf"(?<![a-z0-9])dataset{re.escape(marker)}(?![a-z0-9])"
        )
        return stage_token.search(normalized) is not None or dataset_token.search(
            normalized
        ) is not None

    def visit(
        value: Any,
        trail: tuple[str, ...] = (),
        *,
        reference_field: bool = False,
    ) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                visit(
                    child,
                    (*trail, key),
                    reference_field=key in STAGE_REFERENCE_FIELDS,
                )
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(
                    child,
                    (*trail, str(index)),
                    reference_field=reference_field,
                )
            return
        if not reference_field:
            return
        if not isinstance(value, (str, Path)):
            return
        for marker in sorted(forbidden_markers):
            if mentions_stage(str(value), marker):
                violations.append(f"{'.'.join(trail)}={value!s} references {marker}")

    visit(document)
    if violations:
        joined = "; ".join(violations)
        raise ValueError(f"{context}: cross-stage reference rejected: {joined}")


def _parse_source(name: str, raw: dict[str, Any], dataset_root: Path) -> SourceConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"source {name}: configuration must be a mapping")
    label_source = str(_require(raw, "label_source", f"source {name}"))
    if label_source not in ALLOWED_LABEL_SOURCES:
        raise ValueError(f"source {name}: unsupported label_source {label_source!r}")
    root = _resolve(dataset_root, str(_require(raw, "root", f"source {name}")))
    manifest_value = raw.get("manifest")
    manifest = _resolve(dataset_root, str(manifest_value)) if manifest_value else None
    class_from_parent = _optional_bool(raw, "class_from_parent", False, f"source {name}")
    if label_source == "manifest" and manifest is None:
        raise ValueError(f"source {name}: manifest label_source requires manifest")
    if label_source == "directory" and not class_from_parent:
        raise ValueError(f"source {name}: directory labels require class_from_parent: true")
    if label_source == "none" and (manifest is not None or class_from_parent):
        raise ValueError(f"source {name}: unlabeled source cannot define label metadata")
    if name == "test" and label_source != "none":
        raise ValueError("test source must use label_source: none")
    if name == "train" and label_source == "none":
        raise ValueError("train source must define labels")
    if not root.is_relative_to(dataset_root):
        raise ValueError(f"source {name}: root must be inside dataset_root")
    if manifest is not None and not manifest.is_relative_to(dataset_root):
        raise ValueError(f"source {name}: manifest must be inside dataset_root")
    return SourceConfig(name, root, label_source, manifest, class_from_parent)


def ensure_output_path_is_safe(config: DatasetConfig, path: str | Path) -> Path:
    """Resolve an output and reject every path that could modify source data."""
    output = Path(path).resolve()
    protected_roots = {config.dataset_root, *(source.root for source in config.sources.values())}
    if any(output == root or output.is_relative_to(root) for root in protected_roots):
        raise ValueError(f"output path must be outside dataset and source roots: {output}")
    return output


def load_dataset_config(path: str | Path) -> DatasetConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a mapping")
    raw = raw.get("dataset", raw)
    if not isinstance(raw, dict):
        raise ValueError("dataset configuration must be a mapping")

    competition_stage_value = raw.get("competition_stage")
    competition_stage = (
        None if competition_stage_value is None else str(competition_stage_value).lower()
    )
    if competition_stage is not None and competition_stage not in ALLOWED_COMPETITION_STAGES:
        raise ValueError(
            f"unsupported competition_stage {competition_stage_value!r}; "
            f"expected one of {sorted(ALLOWED_COMPETITION_STAGES)}"
        )
    contract_confirmed = _optional_bool(
        raw, "contract_confirmed", competition_stage is None, "dataset"
    )
    expected_train_count = _optional_positive_int(raw, "expected_train_count", "dataset")
    expected_test_count = _optional_positive_int(raw, "expected_test_count", "dataset")
    expected_num_classes = _optional_positive_int(raw, "expected_num_classes", "dataset")
    if competition_stage is not None:
        validate_competition_stage_references(
            competition_stage, raw, context="dataset configuration"
        )
    if contract_confirmed and competition_stage is not None:
        missing = [
            name
            for name, value in (
                ("expected_train_count", expected_train_count),
                ("expected_test_count", expected_test_count),
                ("expected_num_classes", expected_num_classes),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "confirmed stage-scoped dataset contract is missing: "
                + ", ".join(missing)
            )

    stage = str(_require(raw, "stage", "dataset"))
    if stage not in ALLOWED_STAGES:
        raise ValueError(f"unsupported stage {stage!r}; expected one of {sorted(ALLOWED_STAGES)}")
    base = config_path.parent
    dataset_root = _resolve(base, str(_require(raw, "dataset_root", "dataset")))
    class_mapping = _resolve(dataset_root, str(_require(raw, "class_mapping", "dataset")))
    class_mapping_provenance = str(raw.get("class_mapping_provenance", "official_file"))
    if class_mapping_provenance not in ALLOWED_CLASS_MAPPING_PROVENANCE:
        raise ValueError(
            "unsupported class_mapping_provenance; expected official_file or train_directory_names"
        )
    if class_mapping_provenance == "official_file" and not class_mapping.is_relative_to(dataset_root):
        raise ValueError("official class_mapping must be inside dataset_root")
    extensions_raw = _require(raw, "extensions", "dataset")
    if not isinstance(extensions_raw, list) or not extensions_raw:
        raise ValueError("dataset.extensions must be a non-empty list")
    extensions = tuple(sorted({str(ext).lower() if str(ext).startswith(".") else f".{str(ext).lower()}" for ext in extensions_raw}))

    sources_raw = _require(raw, "sources", "dataset")
    if not isinstance(sources_raw, dict) or "train" not in sources_raw:
        raise ValueError("dataset.sources must contain an explicit train source")
    sources = {name: _parse_source(name, value, dataset_root) for name, value in sources_raw.items()}
    if stage != "synthetic" and "test" not in sources:
        raise ValueError("competition stages must contain an explicit unlabeled test source")
    if class_mapping_provenance == "train_directory_names":
        train_source = sources["train"]
        if train_source.label_source != "directory" or not train_source.class_from_parent:
            raise ValueError(
                "train_directory_names provenance requires directory labels from train parents"
            )
        if class_mapping.is_relative_to(dataset_root):
            raise ValueError("derived class_mapping must be outside the immutable dataset_root")

    audit_raw = _require(raw, "audit", "dataset")
    if not isinstance(audit_raw, dict):
        raise ValueError("dataset.audit must be a mapping")
    audit = AuditConfig(
        output_root=_resolve(base, str(_require(audit_raw, "output_root", "audit"))),
        workers=int(audit_raw.get("workers", 4)),
        near_duplicate_hamming=int(audit_raw.get("near_duplicate_hamming", 3)),
        near_duplicate_max_bucket=int(audit_raw.get("near_duplicate_max_bucket", 500)),
        fail_on_bad_files=_optional_bool(audit_raw, "fail_on_bad_files", True, "audit"),
        fail_on_label_conflicts=_optional_bool(
            audit_raw, "fail_on_label_conflicts", True, "audit"
        ),
    )
    if not 0 <= audit.near_duplicate_hamming <= 3:
        raise ValueError("near_duplicate_hamming must be between 0 and 3 for the LSH guarantee")
    if audit.workers < 1:
        raise ValueError("audit.workers must be at least 1")
    if audit.near_duplicate_max_bucket < 2:
        raise ValueError("near_duplicate_max_bucket must be at least 2")

    split_raw = _require(raw, "split", "dataset")
    if not isinstance(split_raw, dict):
        raise ValueError("dataset.split must be a mapping")
    split = SplitConfig(
        validation_fraction=float(split_raw.get("validation_fraction", 0.2)),
        seed=int(split_raw.get("seed", 42)),
        min_train_per_class=int(split_raw.get("min_train_per_class", 1)),
    )
    if not 0 < split.validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if split.min_train_per_class < 1:
        raise ValueError("min_train_per_class must be at least 1")

    transforms_raw = _require(raw, "transforms", "dataset")
    if not isinstance(transforms_raw, dict):
        raise ValueError("dataset.transforms must be a mapping")
    transforms = TransformConfig(
        image_size=int(transforms_raw.get("image_size", 224)),
        resize_size=int(transforms_raw.get("resize_size", 224)),
        train_crop_scale_min=float(transforms_raw.get("train_crop_scale_min", 0.8)),
        train_horizontal_flip_probability=float(
            transforms_raw.get("train_horizontal_flip_probability", 0.5)
        ),
    )
    if transforms.image_size < 1 or transforms.resize_size < transforms.image_size:
        raise ValueError("transforms require resize_size >= image_size >= 1")
    if not 0 < transforms.train_crop_scale_min <= 1:
        raise ValueError("train_crop_scale_min must be in (0, 1]")
    if not 0 <= transforms.train_horizontal_flip_probability <= 1:
        raise ValueError("train_horizontal_flip_probability must be in [0, 1]")

    submission_raw = raw.get("submission", {})
    if not isinstance(submission_raw, dict):
        raise ValueError("dataset.submission must be a mapping")
    filename_mode = str(submission_raw.get("filename_mode", "basename"))
    if filename_mode not in ALLOWED_SUBMISSION_FILENAME_MODES:
        raise ValueError(
            "submission.filename_mode must be basename or relative_path"
        )
    pattern_value = submission_raw.get("official_class_id_pattern")
    official_class_id_pattern = None
    if pattern_value is not None:
        official_class_id_pattern = str(pattern_value)
        if not official_class_id_pattern:
            raise ValueError(
                "submission.official_class_id_pattern must be null or non-empty"
            )
        try:
            compiled_pattern = re.compile(official_class_id_pattern)
        except re.error as exc:
            raise ValueError(
                "submission.official_class_id_pattern is not a valid regular expression"
            ) from exc
        if compiled_pattern.fullmatch("") is not None:
            raise ValueError(
                "submission.official_class_id_pattern must not match an empty ID"
            )
    submission = SubmissionContract(
        filename_mode=filename_mode,
        official_class_id_pattern=official_class_id_pattern,
    )

    source_roots = [source.root for source in sources.values()]
    if audit.output_root == dataset_root or audit.output_root.is_relative_to(dataset_root):
        raise ValueError("audit output_root cannot be inside dataset_root")
    for source_root in source_roots:
        if audit.output_root == source_root or audit.output_root.is_relative_to(source_root):
            raise ValueError("audit output_root cannot be inside a source root")

    return DatasetConfig(
        config_path=config_path,
        competition_stage=competition_stage,
        contract_confirmed=contract_confirmed,
        expected_train_count=expected_train_count,
        expected_test_count=expected_test_count,
        expected_num_classes=expected_num_classes,
        stage=stage,
        dataset_root=dataset_root,
        class_mapping=class_mapping,
        class_mapping_provenance=class_mapping_provenance,
        extensions=extensions,
        allow_truncated_images=_optional_bool(
            raw, "allow_truncated_images", False, "dataset"
        ),
        sources=sources,
        audit=audit,
        split=split,
        transforms=transforms,
        submission=submission,
    )
