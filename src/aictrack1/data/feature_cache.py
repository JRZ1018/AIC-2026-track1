"""Sharded, provenance-bound frozen feature cache with safe resume support."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .indexed_images import IndexedImage


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FeatureCacheMetadata:
    source: str
    dataset_stage: str
    dataset_config_sha256: str
    manifest_sha256: str
    split_sha256: str | None
    class_mapping_sha256: str
    clip_weight_sha256: str
    clip_upstream_commit: str
    preprocessing: dict[str, Any]
    feature_dim: int
    feature_dtype: str
    normalized: bool
    expected_record_count: int
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class LoadedFeatureCache:
    metadata: FeatureCacheMetadata
    features: torch.Tensor
    labels: torch.Tensor
    sample_indices: torch.Tensor
    sample_ids: tuple[str, ...]
    relative_paths: tuple[str, ...]
    partitions: tuple[str, ...]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FeatureCacheWriter:
    """Append immutable shards and verify the exact record prefix on resume."""

    def __init__(
        self,
        root: str | Path,
        metadata: FeatureCacheMetadata,
        *,
        resume: bool = False,
    ) -> None:
        self.root = Path(root).resolve()
        self.shard_root = self.root / "shards"
        self.metadata = metadata
        metadata_path = self.root / "metadata.json"
        if self.root.exists() and not resume:
            raise FileExistsError(f"feature cache already exists; pass resume=True: {self.root}")
        self.shard_root.mkdir(parents=True, exist_ok=True)
        if metadata_path.exists():
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing != asdict(metadata):
                raise ValueError("feature cache metadata mismatch; refusing unsafe resume")
        else:
            _write_json_atomic(metadata_path, asdict(metadata))
        if (self.root / "complete.json").exists():
            raise ValueError("feature cache is already complete")

    def _shard_paths(self) -> list[Path]:
        paths = sorted(self.shard_root.glob("part-*.pt"))
        expected = [self.shard_root / f"part-{index:06d}.pt" for index in range(len(paths))]
        if paths != expected:
            raise ValueError("feature cache shards are not a contiguous sequence")
        return paths

    @property
    def completed_record_count(self) -> int:
        total = 0
        for path in self._shard_paths():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            total += int(payload["features"].shape[0])
        return total

    def verify_record_prefix(self, records: list[IndexedImage]) -> int:
        offset = 0
        for path in self._shard_paths():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            sample_ids = [str(value) for value in payload["sample_ids"]]
            expected = [record.sample_id for record in records[offset : offset + len(sample_ids)]]
            if sample_ids != expected:
                raise ValueError(f"feature cache record order mismatch at offset {offset}")
            offset += len(sample_ids)
        if offset > len(records):
            raise ValueError("feature cache contains more records than the current selection")
        return offset

    def append(self, features: torch.Tensor, records: list[IndexedImage]) -> Path:
        if features.ndim != 2 or features.shape[1] != self.metadata.feature_dim:
            raise ValueError("feature shard has an unexpected shape")
        if features.shape[0] != len(records) or not records:
            raise ValueError("feature row count must match non-empty records")
        if not torch.isfinite(features).all():
            raise ValueError("feature shard contains non-finite values")
        if self.metadata.normalized:
            norms = torch.linalg.vector_norm(features.float(), dim=1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=2e-3, rtol=2e-3):
                raise ValueError("feature shard is declared normalized but norms differ from one")

        index = len(self._shard_paths())
        output = self.shard_root / f"part-{index:06d}.pt"
        temporary = output.with_suffix(".pt.tmp")
        payload = {
            "features": features.detach().cpu().contiguous(),
            "labels": torch.tensor(
                [-1 if record.internal_class_index is None else record.internal_class_index for record in records],
                dtype=torch.int64,
            ),
            "sample_indices": torch.tensor([record.sample_index for record in records], dtype=torch.int64),
            "sample_ids": [record.sample_id for record in records],
            "relative_paths": [record.relative_path for record in records],
            "partitions": [record.partition for record in records],
        }
        torch.save(payload, temporary)
        os.replace(temporary, output)
        return output

    def finalize(self) -> dict[str, Any]:
        paths = self._shard_paths()
        count = self.completed_record_count
        if count != self.metadata.expected_record_count:
            raise ValueError(
                f"feature cache is incomplete: expected {self.metadata.expected_record_count}, got {count}"
            )
        shards = [
            {"name": path.name, "sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for path in paths
        ]
        complete = {"record_count": count, "shards": shards}
        _write_json_atomic(self.root / "complete.json", complete)
        return complete


def load_feature_cache(root: str | Path, *, require_complete: bool = True) -> LoadedFeatureCache:
    cache_root = Path(root).resolve()
    metadata_raw = json.loads((cache_root / "metadata.json").read_text(encoding="utf-8"))
    metadata = FeatureCacheMetadata(**metadata_raw)
    complete_path = cache_root / "complete.json"
    if require_complete and not complete_path.is_file():
        raise ValueError("feature cache is incomplete")
    shard_paths = sorted((cache_root / "shards").glob("part-*.pt"))
    if not shard_paths:
        raise ValueError("feature cache has no shards")
    expected_paths = [cache_root / "shards" / f"part-{index:06d}.pt" for index in range(len(shard_paths))]
    if shard_paths != expected_paths:
        raise ValueError("feature cache shards are not a contiguous sequence")
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        registered = complete.get("shards", [])
        if [item.get("name") for item in registered] != [path.name for path in shard_paths]:
            raise ValueError("feature cache completion manifest does not match shard names")
        for path, item in zip(shard_paths, registered, strict=True):
            if _sha256_file(path) != item.get("sha256"):
                raise ValueError(f"feature cache shard hash mismatch: {path.name}")

    payloads = [torch.load(path, map_location="cpu", weights_only=True) for path in shard_paths]
    features = torch.cat([payload["features"] for payload in payloads], dim=0)
    labels = torch.cat([payload["labels"] for payload in payloads], dim=0)
    sample_indices = torch.cat([payload["sample_indices"] for payload in payloads], dim=0)
    sample_ids = tuple(str(value) for payload in payloads for value in payload["sample_ids"])
    relative_paths = tuple(str(value) for payload in payloads for value in payload["relative_paths"])
    partitions = tuple(str(value) for payload in payloads for value in payload["partitions"])
    if features.shape != (metadata.expected_record_count, metadata.feature_dim):
        raise ValueError("cached features do not match metadata dimensions")
    expected_dtype = torch.float16 if metadata.feature_dtype == "float16" else torch.float32
    if features.dtype != expected_dtype:
        raise ValueError("cached feature dtype does not match metadata")
    if len(sample_ids) != metadata.expected_record_count:
        raise ValueError("cached identifiers do not match metadata record count")
    if metadata.source == "test" and (labels != -1).any():
        raise ValueError("test feature cache unexpectedly contains labels")
    return LoadedFeatureCache(
        metadata,
        features,
        labels,
        sample_indices,
        sample_ids,
        relative_paths,
        partitions,
    )


def select_feature_partition(cache: LoadedFeatureCache, partition: str) -> tuple[torch.Tensor, torch.Tensor]:
    if cache.metadata.source != "train":
        raise ValueError("training partitions can only be selected from a train cache")
    mask = torch.tensor([value == partition for value in cache.partitions], dtype=torch.bool)
    if not mask.any():
        raise ValueError(f"feature cache contains no {partition!r} records")
    features = cache.features[mask].float()
    labels = cache.labels[mask]
    if (labels < 0).any():
        raise ValueError("training partition contains missing labels")
    return features, labels
