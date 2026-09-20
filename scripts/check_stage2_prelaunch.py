"""Aggregate durable B16 evidence into a Stage 2 prelaunch decision."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.readiness import evaluate_prelaunch
from aictrack1.engine.training_queue import DEFAULT_LOCK_RELATIVE_PATH


TEMPLATES = (
    ROOT / "configs/experiment/stage2_h1_frozen_head.template.yaml",
    ROOT / "configs/experiment/stage2_peft_ce_last4.template.yaml",
    ROOT / "configs/experiment/stage2_peft_ac_last4.template.yaml",
)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _templates_disabled() -> bool:
    for path in TEMPLATES:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        experiment = raw.get("experiment", raw)
        if experiment.get("execution_enabled") is not False:
            return False
    return True


def _repository_clean() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--migration-summary", required=True, type=Path)
    parser.add_argument("--submission-audit", required=True, type=Path)
    parser.add_argument("--capacity-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--account-confirmed",
        action="store_true",
        help="set only after the captain manually verifies qualification and entries",
    )
    parser.add_argument("--require-go", action="store_true")
    args = parser.parse_args()

    decision, checks = evaluate_prelaunch(
        repository_clean=_repository_clean(),
        templates_disabled=_templates_disabled(),
        training_lock_free=not (ROOT / DEFAULT_LOCK_RELATIVE_PATH).exists(),
        migration_smoke=_read_json(args.migration_summary.resolve()),
        submission_simulation=_read_json(args.submission_audit.resolve()),
        capacity=_read_json(args.capacity_report.resolve()),
        account_confirmed=args.account_confirmed,
    )
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema": "aictrack1.stage2_prelaunch.v1",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "decision": decision,
        "git_commit": git_commit,
        "checks": [asdict(check) for check in checks],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if args.require_go and decision != "GO" else 0


if __name__ == "__main__":
    raise SystemExit(main())
