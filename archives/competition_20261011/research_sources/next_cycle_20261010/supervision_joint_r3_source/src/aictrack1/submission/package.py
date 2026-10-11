"""Build and validate the exact single-file competition submission package."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence
from zoneinfo import ZoneInfo

from aictrack1.data.class_mapping import ClassMapping


_REQUIRED_RUN_METADATA = {
    "run_id",
    "checkpoint_sha256",
    "config_hash",
    "data_hash",
    "git_commit",
    "inference_command",
}


@dataclass(frozen=True)
class SubmissionSpec:
    """Explicit format contract; defaults follow the rule PDF example."""

    csv_name: str = "pred_results.csv"
    include_header: bool = False
    filename_column: str = "filename"
    label_column: str = "label"
    filename_mode: str = "basename"
    official_class_id_pattern: str | None = None

    def validate(self) -> None:
        if self.csv_name != "pred_results.csv":
            raise ValueError("competition CSV name must be pred_results.csv")
        if not self.filename_column or not self.label_column:
            raise ValueError("CSV column names must be non-empty")
        if self.filename_column == self.label_column:
            raise ValueError("filename and label columns must be different")
        if self.filename_mode not in {"basename", "relative_path"}:
            raise ValueError("filename_mode must be basename or relative_path")
        if self.official_class_id_pattern is not None:
            if not self.official_class_id_pattern:
                raise ValueError("official_class_id_pattern must be null or non-empty")
            try:
                pattern = re.compile(self.official_class_id_pattern)
            except re.error as exc:
                raise ValueError("official_class_id_pattern is invalid") from exc
            if pattern.fullmatch("") is not None:
                raise ValueError("official_class_id_pattern must not match an empty ID")


@dataclass(frozen=True)
class SubmissionPackage:
    csv_path: Path
    zip_path: Path
    manifest_path: Path
    record_count: int
    csv_sha256: str
    zip_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _normalize_relative_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    if not normalized or any(
        ord(character) < 32 or ord(character) == 127 for character in normalized
    ):
        raise ValueError("test relative paths must be non-empty and contain no control characters")
    parts = normalized.split("/")
    if (
        PurePosixPath(normalized).is_absolute()
        or Path(value).is_absolute()
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"unsafe test relative path: {value!r}")
    return PurePosixPath(normalized).as_posix()


def submission_keys_from_relative_paths(
    relative_paths: Sequence[str],
    *,
    spec: SubmissionSpec = SubmissionSpec(),
) -> tuple[str, ...]:
    """Convert audited test paths into the exact first-column submission keys."""

    spec.validate()
    normalized = tuple(_normalize_relative_path(str(value)) for value in relative_paths)
    if spec.filename_mode == "basename":
        keys = tuple(PurePosixPath(value).name for value in normalized)
    else:
        keys = normalized
    if len({value.casefold() for value in keys}) != len(keys):
        raise ValueError(
            f"test paths collide under submission filename_mode={spec.filename_mode}"
        )
    return keys


def _validate_expected_filenames(
    expected_filenames: Sequence[str],
    spec: SubmissionSpec,
) -> tuple[str, ...]:
    filenames = tuple(str(value) for value in expected_filenames)
    if not filenames:
        raise ValueError("expected filename sequence cannot be empty")
    canonical = submission_keys_from_relative_paths(filenames, spec=spec)
    if canonical != filenames:
        raise ValueError(
            "expected filenames must already use the configured canonical submission format"
        )
    if len(set(filenames)) != len(filenames):
        raise ValueError("expected filename sequence contains duplicates")
    return filenames


def _validate_official_class_ids(
    values: Sequence[str],
    spec: SubmissionSpec,
) -> tuple[str, ...]:
    official_ids = tuple(str(value) for value in values)
    if not official_ids:
        raise ValueError("allowed official class IDs cannot be empty")
    if any(
        not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        for value in official_ids
    ):
        raise ValueError(
            "official class IDs must be non-empty and contain no surrounding whitespace or control characters"
        )
    if len(set(official_ids)) != len(official_ids):
        raise ValueError("allowed official class IDs contain duplicates")
    if spec.official_class_id_pattern is not None:
        pattern = re.compile(spec.official_class_id_pattern)
        invalid = [value for value in official_ids if pattern.fullmatch(value) is None]
        if invalid:
            raise ValueError(
                "official class IDs do not match the configured pattern; "
                f"examples: {invalid[:3]}"
            )
    return official_ids


def _ordered_official_predictions(
    predictions: Mapping[str, int],
    expected_filenames: Sequence[str],
    class_mapping: ClassMapping,
    spec: SubmissionSpec,
) -> list[tuple[str, str]]:
    filenames = _validate_expected_filenames(expected_filenames, spec)
    _validate_official_class_ids(
        [entry.official_class_id for entry in class_mapping.entries], spec
    )
    prediction_names = set(predictions)
    expected_names = set(filenames)
    missing = sorted(expected_names - prediction_names)
    extra = sorted(prediction_names - expected_names)
    if missing or extra:
        raise ValueError(
            f"prediction filename mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )

    rows: list[tuple[str, str]] = []
    for filename in filenames:
        raw_index = predictions[filename]
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise TypeError(f"prediction for {filename!r} must be an integer class index")
        try:
            official = class_mapping.to_official(raw_index)
        except KeyError as exc:
            raise ValueError(f"prediction for {filename!r} has unknown class index {raw_index}") from exc
        rows.append((filename, official))
    return rows


def _parse_csv_rows(raw: bytes, spec: SubmissionSpec) -> list[tuple[str, str]]:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("submission CSV must not contain a UTF-8 BOM")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("submission CSV must be UTF-8") from exc
    if "\r" in text:
        raise ValueError("submission CSV must use LF line endings")
    parsed = list(csv.reader(io.StringIO(text, newline="")))
    if not parsed:
        raise ValueError("submission CSV is empty")
    if spec.include_header:
        expected_header = [spec.filename_column, spec.label_column]
        if parsed[0] != expected_header:
            raise ValueError(f"unexpected CSV header: {parsed[0]!r}")
        parsed = parsed[1:]
    elif parsed[0] == [spec.filename_column, spec.label_column]:
        raise ValueError("submission CSV unexpectedly contains a header")
    if not parsed:
        raise ValueError("submission CSV contains no predictions")
    malformed = [index + 1 + int(spec.include_header) for index, row in enumerate(parsed) if len(row) != 2]
    if malformed:
        raise ValueError(f"submission CSV rows must have exactly two fields: {malformed[:3]}")
    return [(row[0], row[1]) for row in parsed]


def _read_csv_rows(path: Path, spec: SubmissionSpec) -> list[tuple[str, str]]:
    if path.name != spec.csv_name:
        raise ValueError(f"submission CSV must be named {spec.csv_name}")
    return _parse_csv_rows(path.read_bytes(), spec)


def _validate_parsed_rows(
    rows: Sequence[tuple[str, str]],
    expected: Sequence[str],
    allowed: set[str],
) -> None:
    filenames = [filename for filename, _ in rows]
    if len(set(filenames)) != len(filenames):
        raise ValueError("submission CSV contains duplicate filenames")
    if tuple(filenames) != tuple(expected):
        missing = sorted(set(expected) - set(filenames))
        extra = sorted(set(filenames) - set(expected))
        raise ValueError(
            "submission filenames or order do not match the expected sequence: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    for filename, label in rows:
        if label not in allowed:
            raise ValueError(f"invalid official class ID for {filename!r}: {label!r}")


def validate_submission_csv(
    path: str | Path,
    expected_filenames: Sequence[str],
    allowed_official_class_ids: Sequence[str],
    *,
    spec: SubmissionSpec = SubmissionSpec(),
) -> list[tuple[str, str]]:
    """Validate exact order, filenames, labels, encoding, and row structure."""

    spec.validate()
    expected = _validate_expected_filenames(expected_filenames, spec)
    allowed = set(_validate_official_class_ids(allowed_official_class_ids, spec))
    rows = _read_csv_rows(Path(path), spec)
    _validate_parsed_rows(rows, expected, allowed)
    return rows


def validate_submission_zip(
    path: str | Path,
    expected_filenames: Sequence[str],
    allowed_official_class_ids: Sequence[str],
    *,
    spec: SubmissionSpec = SubmissionSpec(),
) -> None:
    """Require one root-level CSV and validate its exact decompressed bytes."""

    archive = Path(path)
    with zipfile.ZipFile(archive, "r") as handle:
        infos = handle.infolist()
        if len(infos) != 1 or infos[0].filename != spec.csv_name or infos[0].is_dir():
            raise ValueError(f"ZIP must contain only root-level {spec.csv_name}")
        if infos[0].flag_bits & 0x1:
            raise ValueError("submission ZIP must not be encrypted")
        payload = handle.read(infos[0])
    spec.validate()
    expected = _validate_expected_filenames(expected_filenames, spec)
    allowed = set(_validate_official_class_ids(allowed_official_class_ids, spec))
    rows = _parse_csv_rows(payload, spec)
    _validate_parsed_rows(rows, expected, allowed)


def build_submission_package(
    predictions: Mapping[str, int],
    expected_filenames: Sequence[str],
    class_mapping: ClassMapping,
    output_dir: str | Path,
    *,
    run_metadata: Mapping[str, object],
    spec: SubmissionSpec = SubmissionSpec(),
) -> SubmissionPackage:
    """Atomically create a validated CSV, deterministic ZIP, and audit manifest."""

    spec.validate()
    missing_metadata = sorted(_REQUIRED_RUN_METADATA - set(run_metadata))
    if missing_metadata:
        raise ValueError(f"run metadata is missing required fields: {missing_metadata}")
    empty_metadata = sorted(
        key for key in _REQUIRED_RUN_METADATA if not str(run_metadata[key]).strip()
    )
    if empty_metadata:
        raise ValueError(f"run metadata fields must be non-empty: {empty_metadata}")
    rows = _ordered_official_predictions(
        predictions, expected_filenames, class_mapping, spec
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / spec.csv_name
    zip_path = output / "pred_results.zip"
    manifest_path = output / "submission_manifest.json"
    collisions = [path for path in (csv_path, zip_path, manifest_path) if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite submission outputs: {collisions}")

    temporary_csv = output / f".{spec.csv_name}.tmp"
    temporary_zip = output / ".pred_results.zip.tmp"
    try:
        with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            if spec.include_header:
                writer.writerow([spec.filename_column, spec.label_column])
            writer.writerows(rows)
        temporary_csv.replace(csv_path)
        allowed = [entry.official_class_id for entry in class_mapping.entries]
        validate_submission_csv(csv_path, expected_filenames, allowed, spec=spec)

        info = zipfile.ZipInfo(spec.csv_name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o600 << 16
        with zipfile.ZipFile(temporary_zip, "w") as archive:
            archive.writestr(info, csv_path.read_bytes())
        temporary_zip.replace(zip_path)
        validate_submission_zip(zip_path, expected_filenames, allowed, spec=spec)

        manifest = {
            "schema_version": 2,
            "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "record_count": len(rows),
            "expected_filenames_sha256": _sequence_sha256(tuple(expected_filenames)),
            "class_mapping_sha256": class_mapping.digest(),
            "csv_sha256": _sha256(csv_path),
            "zip_sha256": _sha256(zip_path),
            "spec": asdict(spec),
            "run_metadata": dict(run_metadata),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return SubmissionPackage(
            csv_path=csv_path,
            zip_path=zip_path,
            manifest_path=manifest_path,
            record_count=len(rows),
            csv_sha256=manifest["csv_sha256"],
            zip_sha256=manifest["zip_sha256"],
        )
    except Exception:
        for path in (temporary_csv, temporary_zip, csv_path, zip_path, manifest_path):
            path.unlink(missing_ok=True)
        raise
