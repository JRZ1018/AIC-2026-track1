from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "docs" / "templates"


def test_stage2_report_has_required_sections_and_no_claimed_result() -> None:
    report = (TEMPLATES / "stage2_technical_report.md").read_text(encoding="utf-8")
    for section in (
        "数据审计与切分",
        "长尾与噪声诊断",
        "实验矩阵与消融",
        "资源与复现",
        "失败实验、限制与风险",
        "复现身份表",
    ):
        assert section in report
    assert "[填]" in report
    assert "61.8016" not in report


def test_chart_schema_keeps_sources_and_missing_value_rules() -> None:
    schema = yaml.safe_load(
        (TEMPLATES / "stage2_chart_schema.yaml").read_text(encoding="utf-8")
    )
    assert schema["schema"] == "aictrack1.stage2_report_charts.v1"
    assert len(schema["charts"]) == 5
    assert all(chart["source"] and chart["required_fields"] for chart in schema["charts"])
    assert any("missing values" in rule for rule in schema["rules"])


def test_material_checklist_covers_identity_and_submission() -> None:
    checklist = (TEMPLATES / "stage2_material_checklist.md").read_text(encoding="utf-8")
    for term in ("environment", "checkpoint", "Git", "CSV", "ZIP", "回退包"):
        assert term in checklist
