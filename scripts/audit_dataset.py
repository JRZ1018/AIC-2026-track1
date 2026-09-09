"""Run a read-only dataset audit from an explicit configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.audit import write_audit_report
from aictrack1.data.contracts import load_dataset_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source", default="train")
    args = parser.parse_args()
    config = load_dataset_config(args.config)
    summary = write_audit_report(config, args.source)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
