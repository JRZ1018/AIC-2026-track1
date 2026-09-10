"""Build and validate the exact single-file competition submission package."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence
from zoneinfo import ZoneInfo

from aictrack1.data.class_mapping import ClassMapping


_FOUR_DIGIT_LABEL = re.compile(r"\d{4}")
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

    def validate(self) -> None:
        if self.csv_name != "pred_results.csv":
            raise ValueError("competition CSV name must be pred_results.csv")
        if not self.filename_column or not self.label_column:
            raise ValueError("CSV column names must be non-empty")
        if self.filename_column == self.label_column:
            raise ValueError("filename and label columns must be different")


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


def _validate_expected_filenames(expected_filenames: Sequence[str]) -> tuple[str, ...]:
    filenames = tuple(str(value) for value in expected_filenames)
    if not filenames:
        raise ValueError("expected filename sequence cannot be empty")
    if any(not value or Path(value).name != value for value in filenames):
        raise ValueError("expected filenames must be non-empty basenames")
    if len(set(filenames)) != len(filenames):
        raise ValueError("expected filename sequence contains duplicates")
    return filenames


def _ordered_official_predictions(
    predictions: Mapping[str, int],
    expected_filenames: Sequence[str],
    class_mapping: ClassMapping,
) -> list[tuple[str, str]]:
    filenames = _validate_expected_filenames(expected_filenames)
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
        if _FOUR_DIGIT_LABEL.fullmatch(official) is None:
            raise ValueError(f"official class ID must be exactly four digits: {official!r}")
        rows.append((filename, official))
    return rows


def _read_csv_rows(path: Path, spec: SubmissionSpec) -> list[tuple[str, str]]:
    if path.name != spec.csv_name:
        raise ValueError(f"submission CSV must be named {spec.csv_name}")
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("submission CSV must not contain a UTF-8 BOM")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("submission CSV must be UTF-8") from exc
    if "\r" in text:
        raise ValueError("submission CSV must use LF line endings")
    with path.open("r", encoding="utf-8", newline="") as handle:
        parsed = list(csv.reader(handle))
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


def validate_submission_csv(
    path: str | Path,
    expected_filenames: Sequence[str],
    allowed_official_class_ids: Sequence[str],
    *,
    spec: SubmissionSpec = SubmissionSpec(),
) -> list[tuple[str, str]]:
    """Validate exact order, filenames, labels, encoding, and row structure."""

    spec.validate()
    expected = _validate_expected_filenames(expected_filenames)
    allowed = set(str(value) for value in allowed_official_class_ids)
    if not allowed or any(_FOUR_DIGIT_LABEL.fullmatch(value) is None for value in allowed):
        raise ValueError("allowed official class IDs must be non-empty four-digit strings")
    rows = _read_csv_rows(Path(path), spec)
    filenames = [filename for filename, _ in rows]
    if len(set(filenames)) != len(filenames):
        raise ValueError("submission CSV contains duplicate filenames")
    if tuple(filenames) != expected:
        missing = sorted(set(expected) - set(filenames))
        extra = sorted(set(filenames) - set(expected))
        raise ValueError(
            "submission filenames or order do not match the expected sequence: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    for filename, label in rows:
        if label not in allowed:
            raise ValueError(f"invalid official class ID for {filename!r}: {label!r}")
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
    with tempfile.TemporaryDirectory(prefix="submission-validate-", dir=archive.parent) as temp:
        extracted = Path(temp) / spec.csv_name
        extracted.write_bytes(payload)
        validate_submission_csv(
            extracted,
            expected_filenames,
            allowed_official_class_ids,
            spec=spec,
        )


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
    rows = _ordered_official_predictions(predictions, expected_filenames, class_mapping)
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
            "schema_version": 1,
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
