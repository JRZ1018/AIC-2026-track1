from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage2_task_order_contains_reproducibility_and_stop_fields() -> None:
    text = (ROOT / "docs/templates/stage2_task_order.md").read_text(encoding="utf-8")
    required = [
        "Git commit（40 位）",
        "配置 SHA-256",
        "train manifest SHA-256",
        "class mapping SHA-256",
        "OpenAI CLIP 权重 SHA-256",
        "唯一命令",
        "唯一输出目录",
        "停止条件",
        "产物清单与哈希",
    ]
    for field in required:
        assert field in text


def test_stage2_operations_has_single_owners_and_escalations() -> None:
    text = (ROOT / "docs/87_复赛单机协作值守与异常升级_20260920.md").read_text(
        encoding="utf-8"
    )
    assert "唯一训练调度者" in text
    assert "B 不点击提交" in text
    for incident in ("官方包损坏", "CUDA OOM", "ACL", "训练崩溃", "平台不可用"):
        assert incident in text
    assert "0:00–0:30" in text
    assert "3:00–4:00" in text
