"""Manifest-backed image datasets with explicit train/validation/test boundaries."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from .contracts import DatasetConfig

# The vocabulary of the frozen project split. Diagnostic views such as the
# long-tail stress split pass a wider set explicitly through allowed_partitions.
DEFAULT_PARTITIONS = frozenset({"train", "validation", "test", "excluded_bad_file"})


@dataclass(frozen=True)
class IndexedImage:
    sample_index: int
    sample_id: str
    relative_path: str
    absolute_path: Path
    source: str
    partition: str
    internal_class_index: int | None
    official_class_id: str | None


def _read_split_assignments(path: Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"sample_id", "split"}.issubset(reader.fieldnames):
            raise ValueError("split file requires sample_id and split columns")
        for row in reader:
            sample_id = str(row["sample_id"])
            if sample_id in assignments:
                raise ValueError(f"duplicate sample_id in split file: {sample_id}")
            assignments[sample_id] = str(row["split"])
    return assignments


def read_indexed_images(
    config: DatasetConfig,
    *,
    source: str,
    manifest_path: str | Path,
    split_path: str | Path | None = None,
    partitions: Iterable[str] | None = None,
    allowed_partitions: Iterable[str] = DEFAULT_PARTITIONS,
    require_complete_split: bool = True,
) -> list[IndexedImage]:
    """Read an audited manifest without inferring labels or partitions from paths.

    ``require_complete_split`` guarantees that the supplied split file covers every
    manifest sample. Diagnostic views that intentionally publish only a subset must
    pass ``False``; the fallback partition is then never selected, because the
    caller filters by explicit partition names.
    """
    if source not in config.sources:
        raise ValueError(f"source {source!r} is not declared in the dataset configuration")
    if source == "train" and split_path is None:
        raise ValueError("train feature extraction requires the fixed split file")
    if source == "test" and split_path is not None:
        raise ValueError("test data cannot receive train/validation partition assignments")

    accepted = frozenset(allowed_partitions)
    if not accepted:
        raise ValueError("allowed_partitions cannot be empty")
    if source == "test" and accepted != DEFAULT_PARTITIONS:
        raise ValueError("test records always use the default partition vocabulary")

    wanted = set(partitions) if partitions is not None else None
    assignments = _read_split_assignments(Path(split_path)) if split_path is not None else {}
    records: list[IndexedImage] = []
    seen: set[str] = set()
    manifest_ids: set[str] = set()
    source_root = config.sources[source].root

    with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_id",
            "relative_path",
            "source_split",
            "internal_class_index",
            "official_class_id",
            "decode_status",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"manifest requires columns {sorted(required)}")
        for manifest_index, row in enumerate(reader):
            sample_id = str(row["sample_id"])
            if sample_id in seen:
                raise ValueError(f"duplicate sample_id in manifest: {sample_id}")
            seen.add(sample_id)
            if str(row["source_split"]) != source:
                raise ValueError(f"manifest sample {sample_id} belongs to another source")
            if str(row["decode_status"]) != "ok":
                raise ValueError(f"manifest sample {sample_id} is not safely decodable")

            relative_path = str(row["relative_path"])
            absolute_path = (source_root / relative_path).resolve()
            if not absolute_path.is_relative_to(source_root):
                raise ValueError(f"sample path escapes source root: {relative_path}")
            if not absolute_path.is_file():
                raise FileNotFoundError(f"audited image is missing: {absolute_path}")

            manifest_ids.add(sample_id)
            partition = assignments.get(sample_id, "test")
            if split_path is not None and sample_id not in assignments:
                if require_complete_split:
                    raise ValueError(f"sample {sample_id} is missing from the fixed split")
                if wanted is not None:
                    continue
            if partition not in accepted:
                raise ValueError(f"unsupported partition {partition!r} for sample {sample_id}")
            if wanted is not None and partition not in wanted:
                continue

            class_value = str(row["internal_class_index"] or "")
            official_value = str(row["official_class_id"] or "")
            if source == "train" and (not class_value or not official_value):
                raise ValueError(f"train sample {sample_id} is missing label metadata")
            if source == "test" and (class_value or official_value):
                raise ValueError("test manifest unexpectedly contains labels")
            records.append(
                IndexedImage(
                    sample_index=manifest_index,
                    sample_id=sample_id,
                    relative_path=relative_path,
                    absolute_path=absolute_path,
                    source=source,
                    partition=partition,
                    internal_class_index=int(class_value) if class_value else None,
                    official_class_id=official_value or None,
                )
            )

    extras = set(assignments) - manifest_ids
    if extras:
        raise ValueError(f"split file contains {len(extras)} unknown sample IDs")
    if not records:
        raise ValueError("selection contains no images")
    return records


class IndexedImageDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        records: list[IndexedImage],
        transform: Callable[[Image.Image], torch.Tensor],
        *,
        allow_truncated_images: bool = False,
        stochastic_seed: int | None = None,
    ) -> None:
        self.records = records
        self.transform = transform
        self.allow_truncated_images = allow_truncated_images
        self.stochastic_seed = stochastic_seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("dataset epoch must be non-negative")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = self.allow_truncated_images
            with Image.open(record.absolute_path) as opened:
                image = opened.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"failed to load audited image {record.absolute_path}") from exc
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous
        if bool(getattr(self.transform, "stochastic", False)):
            if self.stochastic_seed is None:
                raise ValueError("stochastic image transform requires an explicit dataset seed")
            seed = np.random.SeedSequence(
                [self.stochastic_seed, self.epoch, record.sample_index]
            )
            transformed = self.transform(image, rng=np.random.default_rng(seed))
        else:
            transformed = self.transform(image)
        return {
            "image": transformed,
            "label": -1 if record.internal_class_index is None else record.internal_class_index,
            "sample_index": record.sample_index,
            "sample_id": record.sample_id,
            "relative_path": record.relative_path,
            "partition": record.partition,
        }
