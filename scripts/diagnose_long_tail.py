"""Report head/mid/tail, worst-class recall and per-class retention diagnostics.

The script reads only the current-stage training manifest plus the frozen split
file and the long-tail stress split derived from it. It refuses any input that
resolves inside a test directory, runs a single read-only forward pass, and never
writes a submission package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.class_mapping import ClassMapping  # noqa: E402
from aictrack1.data.contracts import load_dataset_config  # noqa: E402
from aictrack1.data.indexed_images import IndexedImageDataset, read_indexed_images  # noqa: E402
from aictrack1.data.long_tail_split import (  # noqa: E402
    LONG_TAIL_EVAL,
    LONG_TAIL_TRAIN,
    read_split_assignments,
    summarize_partition,
)
from aictrack1.data.manifest import read_manifest  # noqa: E402
from aictrack1.data.transforms import build_split_transforms  # noqa: E402
from aictrack1.evaluation.classification import classification_diagnostics  # noqa: E402
from aictrack1.evaluation.long_tail import (  # noqa: E402
    long_tail_diagnostics,
    read_stratification,
    retention_gate,
)
from aictrack1.models.single_model import (  # noqa: E402
    dump_json,
    load_diagnostic_model,
    read_partition_assignments,
    resolve_relative,
    sha256_file,
)

VIEW_COLUMN = "split"


@contextmanager
def _partition_view(tmp_path: Path, assignment: dict[str, str]):
    """Publish a `sample_id,split` view so the audited reader can filter it."""
    unknown = sorted(set(assignment.values()) - {LONG_TAIL_TRAIN, LONG_TAIL_EVAL})
    if unknown:
        raise ValueError(f"unsupported long-tail partitions: {unknown}")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    existed = tmp_path.exists()
    original = tmp_path.read_bytes() if existed else None
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("sample_id,split\n")
            for sample_id in sorted(assignment):
                handle.write(f"{sample_id},{assignment[sample_id]}\n")
        yield tmp_path
    finally:
        if original is None:
            tmp_path.unlink(missing_ok=True)
        else:
            tmp_path.write_bytes(original)


def _assert_not_test(path: Path, *, label: str) -> None:
    parts = [part.lower() for part in path.resolve().parts]
    if "test" in parts:
        raise ValueError(f"{label} resolves inside a test directory: {path}")


def _assignments_hash(assignment: dict[str, str]) -> str:
    payload = "\n".join(f"{key}\t{assignment[key]}" for key in sorted(assignment))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _predict(model, records, transform, *, device: torch.device, batch_size: int, workers: int):
    dataset = IndexedImageDataset(records, transform, allow_truncated_images=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    targets: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
            predictions.extend(int(value) for value in logits.argmax(dim=1).cpu())
            targets.extend(int(value) for value in batch["label"])
    if len(predictions) != len(records):
        raise RuntimeError("inference output count does not match the selected records")
    return torch.tensor(predictions), torch.tensor(targets)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--long-tail-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-balanced-reference", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")

    long_tail_config_path = args.long_tail_config.resolve()
    long_tail_config = yaml.safe_load(long_tail_config_path.read_text(encoding="utf-8"))[
        "long_tail_split"
    ]
    outputs = long_tail_config["outputs"]
    split_path = resolve_relative(long_tail_config_path.parent, outputs["split"])
    strata_path = resolve_relative(long_tail_config_path.parent, outputs["stratification"])
    summary_path = resolve_relative(long_tail_config_path.parent, outputs["summary"])
    for path, label in (
        (split_path, "long-tail split"),
        (strata_path, "stratification"),
        (summary_path, "long-tail summary"),
    ):
        _assert_not_test(path, label=label)
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {label}: run scripts/build_long_tail_split.py first"
            )

    experiment_path = args.experiment_config.resolve()
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))["experiment"]
    dataset_config_path = resolve_relative(experiment_path.parent, experiment["dataset_config"])
    dataset_config = load_dataset_config(dataset_config_path)
    manifest_path = resolve_relative(experiment_path.parent, experiment["manifest"])
    frozen_split_path = resolve_relative(experiment_path.parent, experiment["split"])
    _assert_not_test(manifest_path, label="training manifest")
    _assert_not_test(frozen_split_path, label="frozen split")

    mapping = ClassMapping.from_csv(dataset_config.class_mapping)
    num_classes = len(mapping)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    long_tail_assignment = read_partition_assignments(split_path, "long_tail_split")
    strata = read_stratification(strata_path)
    if strata.num_classes != num_classes:
        raise ValueError("stratification and class mapping disagree")
    long_tail_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if long_tail_summary["num_classes"] != num_classes:
        raise ValueError("long-tail summary and class mapping disagree")

    all_records = read_manifest(manifest_path)
    frozen_assignments = read_split_assignments(frozen_split_path)
    if len(frozen_assignments) != len(all_records):
        raise ValueError("frozen split and manifest cover different sample counts")
    reference_counts = summarize_partition(
        all_records, frozen_assignments, partition="train", num_classes=num_classes
    )
    kept_counts = summarize_partition(
        all_records, long_tail_assignment, partition=LONG_TAIL_TRAIN, num_classes=num_classes
    )

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    fitted = load_diagnostic_model(
        config_path=experiment_path,
        checkpoint_path=args.checkpoint,
        class_mapping_digest=mapping.digest(),
        num_classes=num_classes,
        split_sha256=sha256_file(frozen_split_path),
        device=device,
        git_commit=git_commit,
    )
    transforms = build_split_transforms(dataset_config.transforms)
    evaluation_transform = transforms["validation"]

    def evaluate(records: list, *, label: str) -> dict:
        predictions, targets = _predict(
            fitted.model,
            records,
            evaluation_transform,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        return {
            "partition": label,
            "classification": classification_diagnostics(
                predictions, targets, num_classes=num_classes
            ),
            "long_tail": long_tail_diagnostics(
                predictions,
                targets,
                num_classes=num_classes,
                stratification=strata,
                reference_class_counts=reference_counts,
                evaluated_class_counts=kept_counts,
            ),
        }

    view_path = split_path.parent / "long_tail_diagnostic_view.csv"
    with _partition_view(view_path, long_tail_assignment):
        long_tail_records = read_indexed_images(
            dataset_config,
            source="train",
            manifest_path=manifest_path,
            split_path=view_path,
            partitions=[LONG_TAIL_EVAL],
            allowed_partitions=(LONG_TAIL_TRAIN, LONG_TAIL_EVAL),
            require_complete_split=False,
        )
    if not long_tail_records:
        raise ValueError("long-tail evaluation partition is empty")
    selected = {record.sample_id for record in long_tail_records}
    if not selected.issubset(set(long_tail_assignment)):
        raise RuntimeError("long-tail view returned samples outside the long-tail split")

    report: dict = {
        "scope": (
            "current-stage train manifest, frozen split and deterministic long-tail "
            "stress split only; competition test data not accessed"
        ),
        "diagnostic_model": fitted.provenance(),
        "class_mapping_sha256": mapping.digest(),
        "num_classes": num_classes,
        "long_tail_design": {
            "seed": long_tail_summary["seed"],
            "schedule": long_tail_summary["schedule"],
            "stratification": long_tail_summary["stratification"],
            "partitions": long_tail_summary["partitions"],
            "duplicate_cluster_leakage_count": long_tail_summary[
                "duplicate_cluster_leakage_count"
            ],
            "retention": long_tail_summary["retention"],
            "retention_gate": long_tail_summary["retention_gate"],
            "split_sha256": long_tail_summary["hashes"]["long_tail_split"],
            "assignment_sha256": _assignments_hash(long_tail_assignment),
        },
        "hashes": {
            "experiment_config": fitted.config_sha256,
            "checkpoint": fitted.checkpoint_sha256,
            "train_manifest": sha256_file(manifest_path),
            "frozen_split": sha256_file(frozen_split_path),
        },
        "caveat": (
            "Both partitions are diagnostics only; they never replace the fixed "
            "noisy-validation selection metric. The long-tail evaluation samples come "
            "from the frozen training partition, so this checkpoint has already seen "
            "them; the numbers are an optimistic upper bound and the real long-tail "
            "stress test needs a head trained from scratch on the long-tail train "
            "partition. Per-class retention describes the long-tail schedule, not yet a "
            "training-time sample filter."
        ),
    }
    report["long_tail_eval"] = evaluate(long_tail_records, label="long_tail_eval")
    if not args.skip_balanced_reference:
        validation_records = read_indexed_images(
            dataset_config,
            source="train",
            manifest_path=manifest_path,
            split_path=frozen_split_path,
            partitions=["validation"],
        )
        report["fixed_validation_reference"] = evaluate(
            validation_records, label="fixed_noisy_validation"
        )

    gate_config = long_tail_config["gates"]
    report["retention_gate"] = retention_gate(
        long_tail_summary["retention"],
        min_class_rate=float(gate_config["min_per_class_rate"]),
        min_tail_to_head_ratio=float(gate_config["min_tail_to_head_rate_ratio"]),
    )
    tail_gap = report["long_tail_eval"]["long_tail"]["head_tail_accuracy_gap"]
    report["head_tail_accuracy_gate"] = {
        "max_allowed_gap": float(gate_config["max_tail_to_head_accuracy_gap"]),
        "actual_gap": tail_gap,
        "evaluated": tail_gap is not None,
        "passed": (tail_gap is not None)
        and (tail_gap <= float(gate_config["max_tail_to_head_accuracy_gap"])),
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(output_dir / "long_tail_diagnostics.json", report)
    (output_dir / "summary.md").write_text(_summary_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.4f}%"


def _summary_markdown(report: dict) -> str:
    design = report["long_tail_design"]
    lines = [
        "# 复赛迁移长尾压力与分层诊断",
        "",
        f"- 诊断模型：`{report['diagnostic_model']['run_id']}`",
        f"- 长尾划分 SHA-256：`{design['split_sha256']}`",
        f"- checkpoint SHA-256：`{report['diagnostic_model']['checkpoint_sha256']}`",
        f"- 范围：{report['scope']}",
        f"- 长尾划分重复簇泄漏：{design['duplicate_cluster_leakage_count']}",
        f"- 长尾训练样本：{design['partitions']['long_tail_train']['sample_count']}，"
        f"长尾评估样本：{design['partitions']['long_tail_eval']['sample_count']}",
        "",
    ]
    for key, title in (
        ("fixed_validation_reference", "固定验证集参照（均衡，仅诊断，不作选模）"),
        ("long_tail_eval", "长尾评估划分（仅诊断）"),
    ):
        payload = report.get(key)
        if payload is None:
            continue
        classification = payload["classification"]
        long_tail = payload["long_tail"]
        lines += [
            f"## {title}",
            "",
            f"- 样本数：{classification['sample_count']}，Top-1：{_pct(classification['top1'])}，"
            f"macro：{_pct(classification['macro_accuracy'])}",
            "",
            "| 分层 | 类数 | 评估类数 | 样本数 | 分层准确率 | 类均准确率 | 最差类召回 | 最差类 | 零召回类 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for stratum in ("head", "mid", "tail"):
            item = long_tail["strata"][stratum]
            lines.append(
                f"| {stratum} | {item['class_count']} | {item['evaluated_class_count']} | "
                f"{item['sample_count']} | {_pct(item['accuracy'])} | "
                f"{_pct(item['evaluated_class_mean_accuracy'])} | "
                f"{_pct(item['worst_class_recall'])} | {item['worst_class_index']} | "
                f"{item['zero_recall_class_count']} |"
            )
        lines += [
            "",
            f"- head−tail 准确率差：{_pct(long_tail['head_tail_accuracy_gap'])}；"
            f"head−tail 类均差：{_pct(long_tail['head_tail_macro_gap'])}",
            "",
        ]
    retention = design["retention"]
    lines += [
        "## 每类保留率（长尾调度设计值）",
        "",
        f"- 全体：最小值 {retention['min_rate']:.4f}，均值 {retention['mean_rate']:.4f}，"
        f"饿死类 {retention['starved_class_count']}，仅剩 1 张的类 {retention['minority_class_count']}",
        "",
        "| 分层 | 类数 | 平均保留率 | 最小保留率 | 饿死类数 | 仅剩 1 张的类数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stratum in ("head", "mid", "tail"):
        item = retention["per_stratum"][stratum]
        lines.append(
            f"| {stratum} | {item['class_count']} | {item['mean_rate']:.4f} | "
            f"{item['min_rate']:.4f} | {item['starved_class_count']} | "
            f"{item['minority_class_count']} |"
        )
    lines += [
        "",
        f"- 保留率门禁：{'通过' if report['retention_gate']['passed'] else '未通过'}",
        f"- head−tail 准确率差门禁："
        f"{'通过' if report['head_tail_accuracy_gate']['passed'] else '未通过'}"
        f"（上限 {report['head_tail_accuracy_gate']['max_allowed_gap']:.4f}，"
        f"实测 {_pct(report['head_tail_accuracy_gate']['actual_gap'])}）",
    ]
    unstable = report["long_tail_eval"]["long_tail"]["unstable_strata"]
    if unstable:
        lines.append(
            f"- 支持样本不足、已标记为不可用分层的分层：{', '.join(unstable)}"
            f"（阈值 {report['long_tail_eval']['long_tail']['min_stratum_samples']} 样本）"
        )
    lines += [
        "",
        "## 边界",
        "",
        report["caveat"],
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
