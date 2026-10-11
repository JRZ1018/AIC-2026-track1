"""Serial Stage 2 C8 four-view TTA validation queue for the preregistered P0 gate.

One fold at a time, under the repository atomic training lease held by
``scripts/diagnose_stage2_tta.py``.  Only Stage 2 train-derived validation is
read; no test manifest and no full-train checkpoint can reach this queue.

The promotion thresholds below are the ones frozen in
``docs/124_复赛P0_C8-336同权重双尺度四视图TTA预注册_20260925.md`` before any
fold result existed.  Re-running this module cannot renegotiate them: changing a
constant is a new prospective preregistration, and the historical verdicts of
``docs/115``-``docs/118`` stay untouched either way.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]

# --- Frozen preregistered gate (docs/124). Do not tune after seeing results. ---
TOP1_GAIN_FLOOR_PP = 0.30
TOP1_NO_LOSS_FLOOR_PP = 0.0
MACRO_NO_LOSS_FLOOR_PP = -0.25
TAIL_NO_LOSS_FLOOR_PP = -0.50
MIN_PASSING_FOLDS = 2
# Effective gate for this cross-machine batch is A's original one
# (docs/125 §3): per-fold Top-1 floor + positive group-bootstrap interval +
# macro/tail floors.  A stricter ">=2 folds with positive delta-macro" secondary
# confirmation was added by the user at 2026-09-25 20:57 and withdrawn at
# 23:20, so macro is a floor-only metric here: no direction and no interval
# requirement.  The withdrawal does not change any verdict, because fold2 passes
# under both versions (docs/126 §6).

FOLDS = (
    {
        "tag": "fold1",
        "experiment_config": "configs/experiment/stage2_fold1_c5_336_last8_ac_8ep_seed20260921.yaml",
        "checkpoint": "artifacts/experiments/20260925-stage2-fold1-c5-336-last8-ac-8ep-seed20260921/best.pt",
        "checkpoint_sha256": "dde01c0fff269c0e977a5e79a4a469a1358e1b3e28ca00f85b1e61d1bbb8437e",
        "split_sha256": "c8cd505ad24400546d4e2ac78ba26d313e08b96d4f41e0235960e33da343b03c",
        # Recorded 336 two-view control on this exact checkpoint.  The new run must
        # re-derive it inside the same report, so a drifted pipeline cannot be scored.
        "expected_center_top1": 0.702491819858551,
        "expected_flip_tta_top1": 0.7102599740028381,
    },
    {
        "tag": "fold2",
        "experiment_config": "configs/experiment/stage2_fold2_c5_336_last8_ac_8ep_seed20260921.yaml",
        "checkpoint": "artifacts/experiments/20260925-stage2-fold2-c5-336-last8-ac-8ep-seed20260921/best.pt",
        "checkpoint_sha256": "c4542b1459eedc76ccc64c70cde645f72fd94e6837e9e496420d580f7d28f0b3",
        "split_sha256": "1e5eb7e9adbc234ba933bf76c99896602b8bdce5557879f90224ae2bb4cf2a50",
        "expected_center_top1": 0.7143625617027283,
        "expected_flip_tta_top1": 0.7193731665611267,
    },
    {
        "tag": "fold3",
        "experiment_config": "configs/experiment/stage2_fold3_c5_336_last8_ac_8ep_seed20260921.yaml",
        "checkpoint": "artifacts/experiments/20260925-stage2-fold3-c5-336-last8-ac-8ep-seed20260921/best.pt",
        "checkpoint_sha256": "d005c49221a323d09d6ca659e0c12c964a7f2a089b4eff71184c71c6faf3bd36",
        "split_sha256": "dcef8ccb919fec62d451fe78089d11399e7de2aaf779df7c636cdbdf176cdf68",
        "expected_center_top1": 0.7161112427711487,
        "expected_flip_tta_top1": 0.7236775755882263,
        "overlap_note": (
            "fold3 validation overlaps fold1 by 5,656 (19.02%) and fold2 by "
            "5,946 (20.00%) samples; the three folds are not mutually independent"
        ),
    },
)

EXPECTED_EVALUATED_COUNT = 29737
CONTROL_TOP1_TOLERANCE = 0.0002
BATCH_PROTOCOL = ("scale384", 384)


def fold_outputs(tag: str) -> tuple[Path, Path]:
    """Return (diagnostic directory, paired-bootstrap report) for one fold tag."""

    return (
        ROOT / "artifacts" / "diagnostics" / f"20260925-stage2-c8-{tag}-336-fourview-full",
        ROOT / "artifacts" / "reports" / f"20260925-c8-{tag}-336-fourview-vs-fliptta-bootstrap.json",
    )


def evaluate_fold(
    *,
    tag: str,
    top1_delta_pp: float,
    macro_delta_pp: float,
    tail_delta_pp: float,
    ci95_low_pp: float,
    ci95_high_pp: float,
) -> dict[str, object]:
    """Apply the frozen per-fold thresholds to one paired comparison."""

    top1_gain_ok = top1_delta_pp >= TOP1_GAIN_FLOOR_PP
    no_top1_loss_ok = top1_delta_pp > TOP1_NO_LOSS_FLOOR_PP
    interval_ok = ci95_low_pp > 0.0
    macro_ok = macro_delta_pp >= MACRO_NO_LOSS_FLOOR_PP
    tail_ok = tail_delta_pp >= TAIL_NO_LOSS_FLOOR_PP
    return {
        "tag": tag,
        "top1_delta_pp": top1_delta_pp,
        "macro_delta_pp": macro_delta_pp,
        "tail_delta_pp": tail_delta_pp,
        "group_bootstrap_ci95_pp": [ci95_low_pp, ci95_high_pp],
        "top1_gain_at_least_floor": top1_gain_ok,
        "no_top1_loss": no_top1_loss_ok,
        "bootstrap_interval_positive": interval_ok,
        "macro_within_floor": macro_ok,
        "tail_within_floor": tail_ok,
        "fold_passed": top1_gain_ok and interval_ok and macro_ok and tail_ok,
    }


def evaluate_preregistered_gate(folds: list[dict[str, object]]) -> dict[str, object]:
    """Combine per-fold verdicts into the frozen promotion decision."""

    if not folds:
        raise ValueError("gate evaluation requires at least one fold")
    passing = [fold for fold in folds if fold["fold_passed"]]
    all_positive = all(bool(fold["no_top1_loss"]) for fold in folds)
    all_macro_ok = all(bool(fold["macro_within_floor"]) for fold in folds)
    all_tail_ok = all(bool(fold["tail_within_floor"]) for fold in folds)
    promote = (
        len(passing) >= MIN_PASSING_FOLDS and all_positive and all_macro_ok and all_tail_ok
    )
    failed = [str(fold["tag"]) for fold in folds if not fold["fold_passed"]]
    reasons: list[str] = []
    if len(passing) < MIN_PASSING_FOLDS:
        reasons.append(f"only {len(passing)}/{len(folds)} folds met the full per-fold gate")
    if not all_positive:
        reasons.append("at least one fold lost Top-1 against the two-view control")
    if not all_macro_ok:
        reasons.append("macro accuracy fell past the preregistered floor in some fold")
    if not all_tail_ok:
        reasons.append("tail class-mean recall fell past the preregistered floor in some fold")
    return {
        "thresholds": {
            "top1_gain_floor_pp": TOP1_GAIN_FLOOR_PP,
            "top1_no_loss_floor_pp": TOP1_NO_LOSS_FLOOR_PP,
            "macro_no_loss_floor_pp": MACRO_NO_LOSS_FLOOR_PP,
            "tail_no_loss_floor_pp": TAIL_NO_LOSS_FLOOR_PP,
            "min_passing_folds": MIN_PASSING_FOLDS,
            "macro_requirement": (
                "floor only: delta macro >= -0.25 pp in every fold; no direction and "
                "no interval requirement (A-machine gate, docs/125 section 3)"
            ),
        },
        "folds": folds,
        "passing_folds": [str(fold["tag"]) for fold in passing],
        "failed_folds": failed,
        "promote_candidate": bool(promote),
        "decision": "promote_to_single_test_inference" if promote else "stop_and_report",
        "blocking_reasons": reasons,
        "zero_recall_policy": (
            "per-class zero recall is reported but is not a standalone veto; "
            "the tail and macro floors above are the systematic-collapse guards"
        ),
        "primary_metric_note": (
            "official Top-1 is the primary metric; macro and tail class-mean recall are "
            "reported alongside it as non-collapse guards"
        ),
        "scope": "Stage 2 train-derived fixed validation only; no test access",
    }


def _log(path: Path, message: str) -> None:
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"[{stamp}] {message}\n")


def _diagnostic_complete(directory: Path) -> bool:
    report = directory / "report.json"
    paired = directory / "paired_predictions.csv"
    return report.is_file() and paired.is_file()


def _assert_report_identity(fold: dict[str, str], report: dict[str, object]) -> None:
    if report.get("smoke_only") is not False:
        raise RuntimeError(f"{fold['tag']}: smoke-only report cannot be scored")
    if int(report["evaluated_count"]) != EXPECTED_EVALUATED_COUNT:
        raise RuntimeError(f"{fold['tag']}: incomplete validation coverage")
    if report.get("checkpoint_sha256") != fold["checkpoint_sha256"]:
        raise RuntimeError(f"{fold['tag']}: checkpoint SHA-256 mismatch")
    if report.get("split_sha256") != fold["split_sha256"]:
        raise RuntimeError(f"{fold['tag']}: split SHA-256 mismatch")
    if int(report.get("visual_input_resolution", 0)) != 336:
        raise RuntimeError(f"{fold['tag']}: unexpected visual resolution")
    if int(report.get("four_view_resize", 0)) != BATCH_PROTOCOL[1]:
        raise RuntimeError(f"{fold['tag']}: unexpected four-view resize")
    for key in ("four_view", "flip_tta"):
        if not isinstance(report.get(key), dict):
            raise RuntimeError(f"{fold['tag']}: report is missing the {key} block")
    # The two control branches must reproduce the recorded runs on the same
    # checkpoint; otherwise the extra views are not the only thing that changed.
    for key, expected in (
        ("center", float(fold["expected_center_top1"])),
        ("flip_tta", float(fold["expected_flip_tta_top1"])),
    ):
        observed = float(report[key]["top1"])
        if abs(observed - expected) > CONTROL_TOP1_TOLERANCE:
            raise RuntimeError(
                f"{fold['tag']}: {key} Top-1 {observed:.6f} does not reproduce the "
                f"recorded control {expected:.6f} within {CONTROL_TOP1_TOLERANCE}"
            )


def run_fold(fold: dict[str, str], *, python: str, log_path: Path) -> None:
    directory, bootstrap_path = fold_outputs(fold["tag"])
    if _diagnostic_complete(directory):
        _log(log_path, f"{fold['tag']}: diagnostic already complete, skipping inference")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(
                f"{fold['tag']}: partial diagnostic output at {directory}; "
                "inspect manually instead of overwriting it"
            )
        command = [
            python,
            str(ROOT / "scripts" / "diagnose_stage2_tta.py"),
            "--experiment-config", str(ROOT / fold["experiment_config"]),
            "--checkpoint", str(ROOT / fold["checkpoint"]),
            "--expected-checkpoint-sha256", fold["checkpoint_sha256"],
            "--output-dir", str(directory),
            "--batch-size", "64", "--workers", "4",
            "--four-view-scale384",
        ]
        _log(log_path, f"{fold['tag']}: start four-view validation")
        with (log_path.parent / f"{fold['tag']}.log").open("w", encoding="utf-8", newline="\n") as handle:
            result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"{fold['tag']}: diagnostic failed with exit code {result.returncode}")
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    _assert_report_identity(fold, report)
    if bootstrap_path.is_file():
        _log(log_path, f"{fold['tag']}: paired bootstrap already present, skipping audit")
        return
    audit = [
        python,
        str(ROOT / "scripts" / "audit_stage2_paired_tta_gain.py"),
        "--baseline-predictions", str(directory / "paired_predictions.csv"),
        "--candidate-predictions", str(directory / "paired_predictions.csv"),
        "--baseline-column", "flip_tta_prediction",
        "--candidate-column", "four_view_prediction",
        "--output", str(bootstrap_path),
    ]
    _log(log_path, f"{fold['tag']}: paired near-duplicate group bootstrap")
    with (log_path.parent / f"{fold['tag']}-bootstrap.log").open("w", encoding="utf-8", newline="\n") as handle:
        result = subprocess.run(audit, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{fold['tag']}: paired bootstrap failed with exit code {result.returncode}")


def _fold_verdict(fold: dict[str, str]) -> dict[str, object]:
    directory, bootstrap_path = fold_outputs(fold["tag"])
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    four = report["four_view"]
    two = report["flip_tta"]
    interval = bootstrap["paired_near_duplicate_group_bootstrap"]
    return evaluate_fold(
        tag=fold["tag"],
        top1_delta_pp=(float(four["top1"]) - float(two["top1"])) * 100,
        macro_delta_pp=(float(four["macro_accuracy"]) - float(two["macro_accuracy"])) * 100,
        tail_delta_pp=(
            float(four["tail_class_mean_recall"]) - float(two["tail_class_mean_recall"])
        ) * 100,
        ci95_low_pp=float(interval["ci95_low"]) * 100,
        ci95_high_pp=float(interval["ci95_high"]) * 100,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-dir", type=Path, default=ROOT / "artifacts" / "queues" / "20260925-stage2-c8-fourview-validation")
    parser.add_argument("--analyze-only", action="store_true", help="score existing fold artifacts only")
    parser.add_argument("--skip-bootstrap", action="store_true")
    args = parser.parse_args()
    queue_dir = args.queue_dir.resolve()
    log_path = queue_dir / "chain.log"
    python = sys.executable
    for fold in FOLDS:
        for key in ("experiment_config", "checkpoint"):
            if not (ROOT / fold[key]).is_file():
                raise FileNotFoundError(f"{fold['tag']}: missing {key}")
    if not args.analyze_only:
        _log(log_path, "C8 four-view validation queue started (one fold at a time under the atomic lease)")
        for fold in FOLDS:
            run_fold(fold, python=python, log_path=log_path)
            if args.skip_bootstrap:
                continue
    verdicts = [_fold_verdict(fold) for fold in FOLDS]
    gate = evaluate_preregistered_gate(verdicts)
    gate["protocol"] = {
        "name": "C8",
        "base_checkpoint": "frozen in-fold 336 C5 best.pt per fold; single checkpoint",
        "views": "native336 center, exact flip, resize_short_side384 center_crop336, exact flip",
        "aggregation": "equal-weight mean of four float32 logits",
        "control": "recorded native336 center + exact flip two-view run on the same checkpoint",
    }
    gate["fold_overlap_note"] = FOLDS[2]["overlap_note"]
    output = ROOT / "artifacts" / "reports" / "20260925-c8-336-fourview-preregistered-gate.json"
    if output.exists():
        raise FileExistsError(f"gate report already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _log(log_path, f"gate decision: {gate['decision']} ({gate['blocking_reasons']})")
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
