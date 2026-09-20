from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from simulate_stage2_submission import simulate  # noqa: E402


def test_stage2_scale_submission_is_complete_and_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "simulation"
    result = simulate(output, row_count=37_444, num_classes=750)
    assert result["row_count"] == 37_444
    assert result["unique_filenames"] == 37_444
    assert result["num_classes"] == 750
    assert result["deterministic_csv"] is True
    assert result["deterministic_zip"] is True
    assert result["csv_sha256"] == result["second_csv_sha256"]
    assert result["zip_sha256"] == result["second_zip_sha256"]
    assert result["zip_member"] == "pred_results.csv"
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert audit == result


def test_stage2_submission_simulation_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "simulation"
    output.mkdir()
    try:
        simulate(output, row_count=3, num_classes=2)
    except FileExistsError as exc:
        assert "overwrite" in str(exc)
    else:
        raise AssertionError("simulation must refuse an existing output directory")
