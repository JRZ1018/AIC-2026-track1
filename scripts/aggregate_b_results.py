"""Validate B-worker artifacts and aggregate paired M0/M1 seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.evaluation.b_results import aggregate_batch, check_readiness, write_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-readiness", action="store_true")
    args = parser.parse_args()
    plan = args.plan.resolve()
    if args.check_readiness:
        print(json.dumps(check_readiness(plan, ROOT), indent=2, sort_keys=True))
        return 0
    if args.output is None:
        parser.error("--output is required unless --check-readiness is used")
    report = aggregate_batch(plan, ROOT)
    output = args.output.resolve()
    write_report(report, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
