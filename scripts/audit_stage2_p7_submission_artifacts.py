"""Read-only local audit of the P7 balanced-prior submission package."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import zipfile

import torch

from aictrack1.data.class_mapping import ClassMapping
from aictrack1.submission.package import (
    SubmissionSpec,
    submission_keys_from_relative_paths,
    validate_submission_csv,
    validate_submission_zip,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    package = ROOT / "submissions/20260928_p7_sce_full14_fourview_balanced_a10"
    old_package = ROOT / "submissions/20260928_p5_sce_full14_fourview_a10"
    run = ROOT / "artifacts/experiments/20260928-stage2-p5-sce-336-full-14ep-seed20260921"
    head_run = ROOT / "artifacts/experiments/20260928-stage2-p5-sce-h1-336-full-seed20260921"
    config = ROOT / "configs/experiment/stage2_p5_sce_peft_336_full_14ep_a10.yaml"
    selection_report = ROOT / "artifacts/reports/20260928-p7-balanced-prior-fold2-fold3.json"
    test_manifest = ROOT / "artifacts/audits/stage2/test/samples.csv"
    train_manifest = ROOT / "artifacts/audits/stage2/train/samples.csv"
    split = ROOT / "artifacts/audits/stage2/train/split.csv"
    official_clip = ROOT / "artifacts/weights/ViT-B-32.pt"
    mapping = ClassMapping.from_csv(ROOT / "artifacts/audits/stage2/metadata/classes.csv")
    manifest = json.loads((package / "submission_manifest.json").read_text(encoding="utf-8"))
    selection = json.loads(selection_report.read_text(encoding="utf-8"))
    checkpoint = torch.load(run / "final.pt", map_location="cpu", weights_only=True)
    head = torch.load(head_run / "final.pt", map_location="cpu", weights_only=True)
    spec = SubmissionSpec(**manifest["spec"])
    with test_manifest.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != 37444 or any(row["source_split"] != "test" for row in records):
        raise ValueError("unexpected Stage2 test manifest")
    expected = submission_keys_from_relative_paths(
        [row["relative_path"] for row in records], spec=spec
    )
    allowed = [entry.official_class_id for entry in mapping.entries]
    rows = validate_submission_csv(package / "pred_results.csv", expected, allowed, spec=spec)
    validate_submission_zip(package / "pred_results.zip", expected, allowed, spec=spec)
    with zipfile.ZipFile(package / "pred_results.zip") as archive:
        if archive.read("pred_results.csv") != (package / "pred_results.csv").read_bytes():
            raise ValueError("ZIP CSV bytes differ from the canonical CSV")
    if len(rows) != manifest["record_count"] or len(mapping) != 750:
        raise ValueError("row or class count mismatch")
    run_metadata = manifest["run_metadata"]
    checks = {
        "csv": (package / "pred_results.csv", manifest["csv_sha256"]),
        "zip": (package / "pred_results.zip", manifest["zip_sha256"]),
        "checkpoint": (run / "final.pt", run_metadata["checkpoint_sha256"]),
        "head": (head_run / "final.pt", checkpoint["head_initializer_sha256"]),
        "config": (config, run_metadata["config_hash"]),
        "selection_report": (
            selection_report,
            run_metadata["balanced_prior_selection"]["report_sha256"],
        ),
        "test_manifest": (test_manifest, run_metadata["test_manifest_sha256"]),
        "train_manifest": (train_manifest, checkpoint["train_manifest_sha256"]),
        "split": (split, checkpoint["split_sha256"]),
        "official_clip": (official_clip, checkpoint["clip_weight_sha256"]),
        "inference_source": (
            ROOT / "scripts/infer_and_build_submission.py",
            run_metadata["inference_source_sha256"],
        ),
        "balanced_prior_source": (
            ROOT / "src/aictrack1/evaluation/balanced_prior.py",
            run_metadata["balanced_prior_source_sha256"],
        ),
    }
    hashes = {name: _sha256(path) for name, (path, _) in checks.items()}
    mismatches = {
        name: {"actual": hashes[name], "expected": expected_sha}
        for name, (_, expected_sha) in checks.items()
        if hashes[name] != expected_sha
    }
    if mismatches:
        raise ValueError(f"artifact SHA mismatch: {mismatches}")
    if mapping.digest() != manifest["class_mapping_sha256"]:
        raise ValueError("class mapping digest mismatch")
    balanced = run_metadata["balanced_prior"]
    frozen = run_metadata["balanced_prior_selection"]
    if (
        selection["gate_passed"] is not True
        or selection["selected_candidate"] != "uniform_soft_t0.5_s0.75"
        or frozen["selected_candidate"] != selection["selected_candidate"]
        or balanced["temperature_multiplier"] != 0.5
        or balanced["bias_strength"] != 0.75
    ):
        raise ValueError("balanced prior differs from the frozen train-derived gate")
    if checkpoint["competition_stage"] != "stage2" or checkpoint["num_classes"] != 750:
        raise ValueError("checkpoint Stage2 provenance mismatch")
    if checkpoint["clip_weight_sha256"] != head["clip_weight_sha256"]:
        raise ValueError("head and PEFT official CLIP identity mismatch")
    old_rows = validate_submission_csv(
        old_package / "pred_results.csv", expected, allowed, spec=spec
    )
    changed = sum(old != new for old, new in zip(old_rows, rows, strict=True))
    counts = Counter(label for _, label in rows)
    ordered_counts = sorted(counts.values())
    report = {
        "schema": "aictrack1.stage2_p7_submission_audit.v1",
        "status": "PASS",
        "test_images_decoded_during_audit": 0,
        "model_inference_calls_during_audit": 0,
        "record_count": len(rows),
        "mapping_class_count": len(mapping),
        "predicted_class_count": len(counts),
        "prediction_count_min": ordered_counts[0],
        "prediction_count_median": ordered_counts[len(ordered_counts) // 2],
        "prediction_count_max": ordered_counts[-1],
        "changed_predictions_vs_p5_same_checkpoint_fourview": changed,
        "csv_sha256": hashes["csv"],
        "zip_sha256": hashes["zip"],
        "manifest_sha256": _sha256(package / "submission_manifest.json"),
        "hashes": hashes,
        "balanced_prior": balanced,
        "balanced_prior_selection": frozen,
        "limitations": [
            "No hidden labels are available, so this audit cannot certify online accuracy",
            "Changed prediction count is not a correctness count",
        ],
    }
    output = ROOT / "artifacts/reports/20260928-p7-balanced-submission-audit.json"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
