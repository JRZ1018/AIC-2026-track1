"""Stable manifest construction and serialization."""

from __future__ import annotations

import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
from PIL import Image

from .class_mapping import ClassMapping
from .contracts import DatasetConfig, SourceConfig


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    relative_path: str
    source_split: str
    observed_label: str = ""
    internal_class_index: int | None = None
    official_class_id: str = ""
    file_size: int = 0
    width: int | None = None
    height: int | None = None
    mode: str = ""
    sha256: str = ""
    dhash: str = ""
    decode_status: str = "unchecked"
    duplicate_group: str = ""
    near_duplicate_group: str = ""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_dhash(image: Image.Image) -> str:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.uint8)
    value = 0
    for row in range(8):
        for col in range(8):
            value = (value << 1) | int(pixels[row, col] > pixels[row, col + 1])
    return f"{value:016x}"


def inspect_image(path: Path) -> tuple[int | None, int | None, str, str, str]:
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
            dhash = image_dhash(image)
        return width, height, mode, dhash, "ok"
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        message = str(exc).lower()
        status = "truncated" if "truncated" in message or "broken data stream" in message else "decode_error"
        return None, None, "", "", status


def _manifest_rows(source: SourceConfig) -> list[tuple[str, str]]:
    if source.label_source == "manifest":
        assert source.manifest is not None
        with source.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"relative_path", "observed_label"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"source manifest requires columns {sorted(required)}")
            rows = [(str(row["relative_path"]), str(row["observed_label"])) for row in reader]
    else:
        rows = []
        for path in source.root.rglob("*"):
            if path.is_file():
                label = path.parent.name if source.label_source == "directory" else ""
                rows.append((path.relative_to(source.root).as_posix(), label))
    normalized = [(Path(relative).as_posix(), label) for relative, label in rows]
    if len({relative.casefold() for relative, _ in normalized}) != len(normalized):
        raise ValueError("source contains duplicate relative paths under case-insensitive comparison")
    return sorted(normalized, key=lambda item: item[0].encode("utf-8"))


def build_manifest(config: DatasetConfig, source_name: str) -> list[SampleRecord]:
    if source_name not in config.sources:
        raise KeyError(f"unknown source {source_name!r}")
    source = config.sources[source_name]
    mapping = ClassMapping.from_csv(config.class_mapping)
    def inspect_row(item: tuple[str, str]) -> SampleRecord:
        relative_path, observed_label = item
        path = (source.root / relative_path).resolve()
        if not path.is_relative_to(source.root.resolve()):
            raise ValueError(f"path escapes source root: {relative_path}")
        if path.suffix.lower() not in config.extensions:
            raise ValueError(f"disallowed image extension: {relative_path}")
        sample_id = hashlib.sha256(f"{source_name}\0{relative_path}".encode("utf-8")).hexdigest()
        internal = None
        official = ""
        if source.label_source != "none":
            official = observed_label
            internal = mapping.to_internal(official)
        if not path.is_file():
            return SampleRecord(sample_id, relative_path, source_name, observed_label, internal, official, decode_status="missing")
        width, height, mode, dhash, status = inspect_image(path)
        return SampleRecord(
            sample_id=sample_id,
            relative_path=relative_path,
            source_split=source_name,
            observed_label=observed_label,
            internal_class_index=internal,
            official_class_id=official,
            file_size=path.stat().st_size,
            width=width,
            height=height,
            mode=mode,
            sha256=sha256_file(path),
            dhash=dhash,
            decode_status=status,
        )

    rows = _manifest_rows(source)
    with ThreadPoolExecutor(max_workers=config.audit.workers) as executor:
        return list(executor.map(inspect_row, rows))


def write_manifest(records: list[SampleRecord], path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    names = [field.name for field in fields(SampleRecord)]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return sha256_file(output)


def read_manifest(path: str | Path) -> list[SampleRecord]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        records = []
        for row in reader:
            row["internal_class_index"] = int(row["internal_class_index"]) if row["internal_class_index"] else None
            row["file_size"] = int(row["file_size"])
            row["width"] = int(row["width"]) if row["width"] else None
            row["height"] = int(row["height"]) if row["height"] else None
            records.append(SampleRecord(**row))
    return records


def replace_groups(record: SampleRecord, exact: str, near: str) -> SampleRecord:
    return replace(record, duplicate_group=exact, near_duplicate_group=near)
