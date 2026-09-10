"""Build a strict submission package from already-produced predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.class_mapping import ClassMapping
from aictrack1.data.contracts import ensure_output_path_is_safe, load_dataset_config
from aictrack1.submission import SubmissionSpec, build_submission_package


def _load_predictions(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "internal_class_index"}
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise ValueError(
                "prediction input must have exactly filename,internal_class_index columns"
            )
        predictions: dict[str, int] = {}
        for line_number, row in enumerate(reader, start=2):
            filename = str(row["filename"])
            if filename in predictions:
                raise ValueError(f"duplicate prediction filename at line {line_number}: {filename}")
            predictions[filename] = int(row["internal_class_index"])
    return predictions


def _load_expected_filenames(path: Path) -> list[str]:
    return [line for raw in path.read_text(encoding="utf-8-sig").splitlines() if (line := raw.strip())]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--predictions-csv", required=True, type=Path)
    parser.add_argument("--expected-filenames", required=True, type=Path)
    parser.add_argument("--run-metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--include-header",
        action="store_true",
        help="Non-default compatibility switch; only use if the organizer confirms a header.",
    )
    args = parser.parse_args()

    dataset = load_dataset_config(args.dataset_config)
    output = ensure_output_path_is_safe(dataset, args.output_dir)
    class_mapping = ClassMapping.from_csv(dataset.class_mapping)
    predictions = _load_predictions(args.predictions_csv.resolve())
    expected = _load_expected_filenames(args.expected_filenames.resolve())
    run_metadata = json.loads(args.run_metadata.read_text(encoding="utf-8"))
    if not isinstance(run_metadata, dict):
        raise ValueError("run metadata JSON must contain an object")

    package = build_submission_package(
        predictions,
        expected,
        class_mapping,
        output,
        run_metadata=run_metadata,
        spec=SubmissionSpec(include_header=args.include_header),
    )
    print(
        json.dumps(
            {
                "csv_path": str(package.csv_path),
                "zip_path": str(package.zip_path),
                "manifest_path": str(package.manifest_path),
                "record_count": package.record_count,
                "csv_sha256": package.csv_sha256,
                "zip_sha256": package.zip_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
