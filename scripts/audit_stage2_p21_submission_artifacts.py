"""Independent, read-only audit of the user-requested P21 exploratory package."""

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
    SubmissionSpec, submission_keys_from_relative_paths,
    validate_submission_csv, validate_submission_zip,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "submissions/20261001_p21_fold2_20ep_fourview_balanced_exploratory"
RUN = ROOT / "artifacts/experiments/20261001-stage2-p21-fold2-sce-ac-20ep-4090-screen"
HEAD = ROOT / "artifacts/experiments/20260925-stage2-fold2-c5-h1-336-seed20260921/best.pt"
CONFIG = ROOT / "configs/experiment/stage2_p21_fold2_sce_ac_20ep_4090_screen.yaml"
REPORT = ROOT / "artifacts/reports/20260928-p7-balanced-prior-fold2-fold3.json"
TEST = ROOT / "artifacts/audits/stage2/test/samples.csv"
TRAIN = ROOT / "artifacts/audits/stage2/train/samples.csv"
SPLIT = ROOT / "artifacts/audits/stage2/train/split_fold2_seeded_random_20260924.csv"
MAPPING = ROOT / "artifacts/audits/stage2/metadata/classes.csv"
CLIP = ROOT / "artifacts/weights/ViT-B-32.pt"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads((PACKAGE / "submission_manifest.json").read_text(encoding="utf-8"))
    run = manifest["run_metadata"]
    checkpoint = torch.load(RUN / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("competition_stage") != "stage2" or checkpoint.get("num_classes") != 750:
        raise ValueError("P21 checkpoint Stage2 provenance is invalid")
    metrics = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    if (len(metrics["history"]) != 20 or metrics["history"][-1]["epoch"] != 20
            or metrics["selected_checkpoint_sha256"] != run["checkpoint_sha256"]):
        raise ValueError("P21 model is not the complete frozen twenty-epoch candidate")
    mapping = ClassMapping.from_csv(MAPPING)
    spec = SubmissionSpec(**manifest["spec"])
    with TEST.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    if len(records) != 37444 or any(row["source_split"] != "test" for row in records):
        raise ValueError("official Stage2 test manifest contract failed")
    filenames = submission_keys_from_relative_paths([row["relative_path"] for row in records], spec=spec)
    allowed = [entry.official_class_id for entry in mapping.entries]
    rows = validate_submission_csv(PACKAGE / "pred_results.csv", filenames, allowed, spec=spec)
    validate_submission_zip(PACKAGE / "pred_results.zip", filenames, allowed, spec=spec)
    with zipfile.ZipFile(PACKAGE / "pred_results.zip") as archive:
        if archive.namelist() != ["pred_results.csv"]:
            raise ValueError("ZIP has unexpected files")
        if archive.read("pred_results.csv") != (PACKAGE / "pred_results.csv").read_bytes():
            raise ValueError("ZIP and standalone CSV differ")
    if len(rows) != 37444 or len(mapping) != 750 or manifest["record_count"] != 37444:
        raise ValueError("submission count or class mapping mismatch")
    expected = {
        "csv": (PACKAGE / "pred_results.csv", manifest["csv_sha256"]),
        "zip": (PACKAGE / "pred_results.zip", manifest["zip_sha256"]),
        "checkpoint": (RUN / "best.pt", run["checkpoint_sha256"]),
        "head": (HEAD, checkpoint["head_initializer_sha256"]),
        "config": (CONFIG, run["config_hash"]),
        "selection_report": (REPORT, run["balanced_prior_selection"]["report_sha256"]),
        "test_manifest": (TEST, run["test_manifest_sha256"]),
        "train_manifest": (TRAIN, checkpoint["train_manifest_sha256"]),
        "split": (SPLIT, checkpoint["split_sha256"]),
        "official_clip": (CLIP, checkpoint["clip_weight_sha256"]),
        "inference_source": (ROOT / "scripts/infer_and_build_submission.py", run["inference_source_sha256"]),
        "balanced_prior_source": (ROOT / "src/aictrack1/evaluation/balanced_prior.py", run["balanced_prior_source_sha256"]),
    }
    hashes = {name: sha(path) for name, (path, _) in expected.items()}
    wrong = {name: {"actual": hashes[name], "expected": digest}
             for name, (_, digest) in expected.items() if hashes[name] != digest}
    if wrong or mapping.digest() != manifest["class_mapping_sha256"]:
        raise ValueError(f"artifact SHA/class mapping mismatch: {wrong}")
    selected = json.loads(REPORT.read_text(encoding="utf-8"))
    if (selected["selected_candidate"] != "uniform_soft_t0.5_s0.75"
            or selected["gate_passed"] is not True
            or run["balanced_prior"]["temperature_multiplier"] != 0.5
            or run["balanced_prior"]["bias_strength"] != 0.75):
        raise ValueError("inference-only prior was not the previously frozen P7 choice")
    counts = Counter(label for _, label in rows)
    old = ROOT / "submissions/20260928_p7_sce_full14_fourview_balanced_a10/pred_results.csv"
    old_rows = validate_submission_csv(old, filenames, allowed, spec=spec)
    changed = sum(a != b for a, b in zip(old_rows, rows, strict=True))
    summary = {
        "schema": "aictrack1.stage2_p21_exploratory_submission_audit.v1",
        "status": "PASS", "record_count": len(rows), "mapping_class_count": len(mapping),
        "predicted_class_count": len(counts), "changed_predictions_vs_p7": changed,
        "csv_sha256": hashes["csv"], "zip_sha256": hashes["zip"],
        "manifest_sha256": sha(PACKAGE / "submission_manifest.json"), "hashes": hashes,
        "limitations": ["Fold2 model used only 118,958 Stage2 training images; no full-data retrain",
                        "Matched fourteen-epoch P21 control was not run before this user-requested package",
                        "No hidden labels were used or available; online gain is unproven"],
    }
    destination = ROOT / "artifacts/reports/20261001-p21-exploratory-submission-audit.json"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
