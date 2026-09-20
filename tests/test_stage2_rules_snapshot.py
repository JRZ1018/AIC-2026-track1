from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rules_snapshot_distinguishes_public_facts_from_account_checks() -> None:
    text = (ROOT / "docs/88_复赛平台账号与规则就绪快照_20260920.md").read_text(
        encoding="utf-8"
    )
    assert "148,695 train" in text
    assert "37,444 test" in text
    assert "提交格式与初赛一致" in text
    assert "待人工核验" in text
    assert "没有读取、填写或保存账号" in text


def test_stage2_submission_contract_is_only_partially_confirmed() -> None:
    questions = (ROOT / "docs/compliance_questions.md").read_text(encoding="utf-8")
    assert "C03" in questions and "部分确认" in questions
    assert "C19" in questions and "test 层级" in questions
