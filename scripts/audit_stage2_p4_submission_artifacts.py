"""Read-only audit of the fixed P4 package; never runs image inference."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
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


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit() -> dict:
    torch.set_num_threads(2)
    package = ROOT / "submissions/20260927_p4_fallback_gce_q07_336_full_14ep_fourview"
    manifest = read_json(package / "submission_manifest.json")
    run = ROOT / "artifacts/experiments/20260927-stage2-p4-fallback-gce-q07-336-full-14ep-seed20260927"
    head_run = ROOT / "artifacts/experiments/20260927-stage2-p4-fallback-h1-336-full-seed20260927"
    metrics = read_json(run / "metrics.json")
    head_metrics = read_json(head_run / "metrics.json")
    checkpoint = torch.load(run / "final.pt", map_location="cpu", weights_only=True)
    head = torch.load(head_run / "final.pt", map_location="cpu", weights_only=True)
    mapping = ClassMapping.from_csv(ROOT / "artifacts/audits/stage2/metadata/classes.csv")
    spec = SubmissionSpec(**manifest["spec"])
    test_manifest = ROOT / "artifacts/audits/stage2/test/samples.csv"
    with test_manifest.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 37444 and all(r["source_split"] == "test" for r in records)
    expected = submission_keys_from_relative_paths([r["relative_path"] for r in records], spec=spec)
    allowed = [entry.official_class_id for entry in mapping.entries]
    rows = validate_submission_csv(package / "pred_results.csv", expected, allowed, spec=spec)
    validate_submission_zip(package / "pred_results.zip", expected, allowed, spec=spec)
    with zipfile.ZipFile(package / "pred_results.zip") as archive:
        assert archive.read("pred_results.csv") == (package / "pred_results.csv").read_bytes()
    assert len(mapping) == 750 and len(rows) == manifest["record_count"] == 37444
    assert hashlib.sha256("\n".join(expected).encode()).hexdigest() == manifest["expected_filenames_sha256"]
    assert mapping.digest() == manifest["class_mapping_sha256"] == checkpoint["class_mapping_sha256"]
    checks = {
        "csv": (package / "pred_results.csv", manifest["csv_sha256"]),
        "zip": (package / "pred_results.zip", manifest["zip_sha256"]),
        "checkpoint": (run / "final.pt", manifest["run_metadata"]["checkpoint_sha256"]),
        "head": (head_run / "final.pt", checkpoint["head_initializer_sha256"]),
        "config": (ROOT / "configs/experiment/stage2_p4_fallback_peft_336_full_14ep.yaml", checkpoint["config_sha256"]),
        "test_manifest": (test_manifest, manifest["run_metadata"]["test_manifest_sha256"]),
        "train_manifest": (ROOT / "artifacts/audits/stage2/train/samples.csv", checkpoint["train_manifest_sha256"]),
        "split": (ROOT / "artifacts/audits/stage2/train/split.csv", checkpoint["split_sha256"]),
        "official_clip": (ROOT / "artifacts/weights/ViT-B-32.pt", checkpoint["clip_weight_sha256"]),
    }
    hashes = {}
    for name, (path, expected_sha) in checks.items():
        hashes[name] = sha(path)
        assert hashes[name] == expected_sha, name
    for key in ("competition_stage", "num_classes", "class_mapping_sha256", "train_manifest_sha256", "split_sha256", "clip_weight_sha256"):
        assert checkpoint[key] == head[key] == metrics["checkpoint_provenance"][key], key
    assert checkpoint["competition_stage"] == "stage2" and checkpoint["num_classes"] == 750
    assert checkpoint["normalize_features"] is head["feature_cache_metadata"]["normalized"] is True
    assert head["feature_cache_metadata"]["source"] == "train"
    assert head["feature_cache_metadata"]["training_record_count"] == 148695
    assert checkpoint["mode"] == "last_n_blocks"
    assert checkpoint["visual_input_resolution"] == 336
    assert checkpoint["position_interpolation"] == "bicubic_7x7_to_10x10"
    assert checkpoint["selection_rule"] == "fixed final epoch 14; no validation selection"
    assert head["selection_rule"] == "fixed final epoch 100; no validation selection"
    for model_metrics, epochs in ((metrics, 14), (head_metrics, 100)):
        assert model_metrics["train_count"] == 148695 and model_metrics["validation_count"] == 0
        assert [r["epoch"] for r in model_metrics["history"]] == list(range(1, epochs + 1))
        assert model_metrics["training_scope"] == "full_train_fixed_epochs"
    assert metrics["final_checkpoint_sha256"] == hashes["checkpoint"]
    assert head_metrics["final_checkpoint_sha256"] == hashes["head"]
    decisions = [r["confidence_weighting"]["next_effective_quality"]["decision"] for r in metrics["history"]]
    assert decisions == ["L1"] * 14 and metrics["early_stop"] is None
    state = checkpoint["trainable_state"]
    assert len(state) == 96
    assert {int(key.split(".")[3]) for key in state} == set(range(4, 12))
    assert all(key.startswith("visual.transformer.resblocks.") for key in state)
    assert set(head["model"]) == {"linear.weight", "linear.bias"}
    assert tuple(head["model"]["linear.weight"].shape) == (750, 512)
    assert tuple(head["model"]["linear.bias"].shape) == (750,)
    assert all(torch.isfinite(value).all().item() for value in [*state.values(), *head["model"].values()])
    source_files = (
        "scripts/infer_and_build_submission.py", "src/aictrack1/evaluation/four_view.py",
        "src/aictrack1/data/transforms.py", "src/aictrack1/engine/peft.py",
        "src/aictrack1/models/peft.py", "src/aictrack1/models/heads.py",
        "src/aictrack1/models/clip_resolution.py", "src/aictrack1/models/clip_vit_b32.py",
        "src/aictrack1/submission/package.py", "src/aictrack1/data/class_mapping.py",
    )
    return {
        "audit_time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "PASS", "test_images_decoded": 0, "model_inference_calls": 0,
        "row_count": len(rows), "mapping_class_count": len(mapping),
        "predicted_class_count": len({label for _, label in rows}),
        "peft_epochs": 14, "h1_epochs": 100, "ac_decisions": decisions,
        "hashes": hashes, "manifest_sha256": sha(package / "submission_manifest.json"),
        "current_source_sha256_not_historical_inference_snapshot": {p: sha(ROOT / p) for p in source_files},
        "limitations": ["Does not certify hidden-label accuracy", "No original per-image logits were archived", "Current source hashes do not prove historical dirty-tree source identity"],
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, ensure_ascii=False))
