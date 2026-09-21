"""Reject exact file-content overlap between audited train and test manifests."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.manifest import read_manifest  # noqa: E402
from aictrack1.data.overlap import exact_cross_source_overlap  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--test-manifest", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    summary, rows = exact_cross_source_overlap(
        read_manifest(args.train_manifest), read_manifest(args.test_manifest)
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["sha256", "train_count", "test_count", "train_paths", "test_paths"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if rows:
        raise RuntimeError(
            f"train/test exact-content overlap detected: {len(rows)} SHA-256 groups"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
