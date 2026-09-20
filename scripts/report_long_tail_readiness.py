"""Generate the pre-registered L0/L1/L2 report from an audited train manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.class_mapping import ClassMapping  # noqa: E402
from aictrack1.data.manifest import read_manifest  # noqa: E402
from aictrack1.evaluation.long_tail_readiness import (  # noqa: E402
    support_distribution_profile,
)


def _markdown(report: dict[str, object]) -> str:
    support = report["support"]
    assert isinstance(support, dict)
    triggers = report["triggers"]
    assert isinstance(triggers, list)
    lines = [
        "# 长尾就绪判定",
        "",
        f"- 决策级别：**{report['decision']}**",
        f"- 类别数：{report['class_count']}",
        f"- 样本数：{report['sample_count']}",
        f"- 零样本类：{len(report['zero_sample_classes'])}",
        f"- 无法安全切分类：{len(report['unsafe_split_classes'])}",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
    ]
    for key in ("min", "p1", "p10", "median", "p90", "max", "max_to_min", "p10_to_median", "gini", "effective_number"):
        lines.append(f"| {key} | {support[key]} |")
    lines += ["", "## 触发项", ""]
    if not triggers:
        lines.append("- 无：保持 L0，只启用诊断与自然采样。")
    else:
        for item in triggers:
            lines.append(
                f"- {item['level']} / {item['metric']}：actual={item['actual']}，"
                f"threshold={item['threshold']}"
            )
    lines += [
        "",
        "## 边界",
        "",
        "该判定仅使用当前阶段 train 标签支持数；不得读取 test 预测或分布，也不自动启用长尾算法。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--class-mapping", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--min-train-per-class", type=int, default=1)
    args = parser.parse_args()
    mapping = ClassMapping.from_csv(args.class_mapping)
    counts = [0] * len(mapping.entries)
    for record in read_manifest(args.manifest):
        if record.source_split != "train":
            raise ValueError("long-tail readiness accepts train manifest rows only")
        if record.internal_class_index is None:
            raise ValueError("train manifest row is missing internal_class_index")
        counts[record.internal_class_index] += 1
    report = support_distribution_profile(
        counts,
        validation_fraction=args.validation_fraction,
        min_train_per_class=args.min_train_per_class,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
