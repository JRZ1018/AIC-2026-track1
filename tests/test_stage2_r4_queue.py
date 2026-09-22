from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_queue_module():
    path = ROOT / "scripts" / "run_stage2_r4_ce_queue.py"
    spec = importlib.util.spec_from_file_location("stage2_r4_queue", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r4_queue_contains_only_the_two_unlocked_ce_screens() -> None:
    module = _load_queue_module()
    assert [item.tag for item in module.QUEUE] == ["R4-2", "R4-3"]
    assert all("stage2_r4_ce_" in item.config for item in module.QUEUE)
    assert all("ac_" not in item.config for item in module.QUEUE)
    assert module.LOGS.name == "20260922-stage2-r4-ce"


def test_r4_ac_queue_contains_only_the_two_unlocked_protection_screens() -> None:
    path = ROOT / "scripts" / "run_stage2_r4_ac_queue.py"
    spec = importlib.util.spec_from_file_location("stage2_r4_ac_queue", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert [item.tag for item in module.QUEUE] == ["R4-4", "R4-5"]
    assert all("stage2_r4_ac_" in item.config for item in module.QUEUE)
    assert module.LOGS.name == "20260922-stage2-r4-ac"
