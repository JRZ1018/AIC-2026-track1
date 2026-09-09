"""Build a deterministic data manifest without modifying source files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.contracts import ensure_output_path_is_safe, load_dataset_config
from aictrack1.data.manifest import build_manifest, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source", default="train")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = load_dataset_config(args.config)
    records = build_manifest(config, args.source)
    output = ensure_output_path_is_safe(config, args.output)
    digest = write_manifest(records, output)
    print(f"samples={len(records)} sha256={digest} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
