"""Generate the configured procedural noisy-label fixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.synthetic import generate_synthetic_dataset, synthetic_config_from_mapping


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    path = args.config.resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = synthetic_config_from_mapping(raw, path.parent)
    summary = generate_synthetic_dataset(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
