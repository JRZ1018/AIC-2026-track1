"""Inspect or explicitly clear the repository-wide training queue lock."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import (  # noqa: E402
    DEFAULT_LOCK_RELATIVE_PATH,
    clear_stale_training_lock,
    read_training_lock,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("status", "clear"))
    parser.add_argument(
        "--confirm-stale-token",
        help="exact token printed by status; required for clear",
    )
    args = parser.parse_args()
    lock_path = ROOT / DEFAULT_LOCK_RELATIVE_PATH
    record = read_training_lock(lock_path)
    if args.action == "status":
        if record is None:
            print(json.dumps({"status": "unlocked", "path": str(lock_path)}, indent=2))
        else:
            print(
                json.dumps(
                    {"status": "locked", "path": str(lock_path), "record": asdict(record)},
                    indent=2,
                    sort_keys=True,
                )
            )
        return 0
    if record is None:
        raise FileNotFoundError(f"training lock does not exist: {lock_path}")
    if not args.confirm_stale_token:
        raise ValueError("clear requires --confirm-stale-token from a fresh status check")
    removed = clear_stale_training_lock(
        lock_path,
        confirmed_token=args.confirm_stale_token,
    )
    print(
        json.dumps(
            {"status": "cleared", "path": str(lock_path), "record": asdict(removed)},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
