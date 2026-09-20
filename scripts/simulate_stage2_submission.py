"""Build two deterministic 37,444-row synthetic Stage 2 submission packages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.class_mapping import ClassEntry, ClassMapping  # noqa: E402
from aictrack1.submission.package import (  # noqa: E402
    SubmissionSpec,
    build_submission_package,
    validate_submission_csv,
    validate_submission_zip,
)


def simulate(output: Path, *, row_count: int, num_classes: int) -> dict[str, object]:
    if row_count < 1 or num_classes < 1:
        raise ValueError("row_count and num_classes must be positive")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite simulation output: {output}")
    output.mkdir(parents=True)
    mapping = ClassMapping(
        [ClassEntry(f"cls_{index:03d}", index) for index in range(num_classes)]
    )
    filenames = tuple(
        f"batch-{index % 17:02d}/sample-{index:05d}.jpg" for index in range(row_count)
    )
    predictions = {filename: index % num_classes for index, filename in enumerate(filenames)}
    metadata = {
        "run_id": "stage2-synthetic-package-readiness",
        "checkpoint_sha256": "1" * 64,
        "config_hash": "2" * 64,
        "data_hash": "3" * 64,
        "git_commit": "4" * 40,
        "inference_command": "synthetic readiness only; no image inference",
    }
    spec = SubmissionSpec(
        filename_mode="relative_path",
        official_class_id_pattern=r"cls_\d{3}",
    )
    first = build_submission_package(
        predictions,
        filenames,
        mapping,
        output / "package-a",
        run_metadata=metadata,
        spec=spec,
    )
    second = build_submission_package(
        predictions,
        filenames,
        mapping,
        output / "package-b",
        run_metadata=metadata,
        spec=spec,
    )
    allowed = [entry.official_class_id for entry in mapping.entries]
    validate_submission_csv(first.csv_path, filenames, allowed, spec=spec)
    validate_submission_zip(first.zip_path, filenames, allowed, spec=spec)
    if first.csv_sha256 != second.csv_sha256 or first.zip_sha256 != second.zip_sha256:
        raise RuntimeError("synthetic submission packaging is not byte-deterministic")
    result = {
        "schema": "aictrack1.stage2_submission_simulation.v1",
        "scope": "synthetic manifest and predictions only; no competition test data",
        "row_count": row_count,
        "num_classes": num_classes,
        "unique_filenames": len(set(filenames)),
        "filename_mode": spec.filename_mode,
        "official_class_id_pattern": spec.official_class_id_pattern,
        "csv_sha256": first.csv_sha256,
        "zip_sha256": first.zip_sha256,
        "second_csv_sha256": second.csv_sha256,
        "second_zip_sha256": second.zip_sha256,
        "deterministic_csv": first.csv_sha256 == second.csv_sha256,
        "deterministic_zip": first.zip_sha256 == second.zip_sha256,
        "zip_member": spec.csv_name,
    }
    (output / "audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--row-count", type=int, default=37_444)
    parser.add_argument("--num-classes", type=int, default=750)
    args = parser.parse_args()
    result = simulate(args.output.resolve(), row_count=args.row_count, num_classes=args.num_classes)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
