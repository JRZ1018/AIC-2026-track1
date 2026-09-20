"""Rebuild and validate the canonical experiment index from run artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.experiment_index import (  # noqa: E402
    read_index_text,
    rebuild_index,
    write_index,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=ROOT / "experiments" / "index.csv")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--verify-checkpoints", action="store_true")
    parser.add_argument(
        "--allow-legacy-discovery",
        action="store_true",
        help="one-time migration only: permit newly discovered historical runs without identities",
    )
    parser.add_argument(
        "--preserve-ledger-fields-from-git-ref",
        help="restore established data/split/config/git fields from a Git ref before rebuilding",
    )
    args = parser.parse_args()
    baseline_rows = None
    if args.preserve_ledger_fields_from_git_ref:
        completed = subprocess.run(
            [
                "git",
                "show",
                f"{args.preserve_ledger_fields_from_git_ref}:experiments/index.csv",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8-sig",
        )
        baseline_rows = read_index_text(completed.stdout)
    rows, audit = rebuild_index(
        ROOT,
        args.index,
        verify_checkpoints=args.verify_checkpoints,
        baseline_rows=baseline_rows,
        allow_legacy_discovery=args.allow_legacy_discovery,
    )
    if args.write:
        write_index(args.index, rows)
    payload = asdict(audit)
    payload["mode"] = "written" if args.write else "check_only"
    payload["index"] = str(args.index.resolve())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
