from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from aictrack1.evaluation.b_results import aggregate_batch, check_readiness


MANIFEST_SHA = "a" * 64
SPLIT_SHA = "b" * 64
COMMIT = "c" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_run(
    root: Path,
    *,
    run_id: str,
    pair_id: str,
    role: str,
    seed: int,
    top1: float,
) -> dict:
    config_path = root / "configs" / f"{run_id}.yaml"
    result_dir = root / "artifacts" / run_id
    config_path.parent.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": {
            "run_id": run_id,
            "output_dir": f"../../artifacts/{run_id}",
            "training": {"seed": seed},
        }
    }
    config_text = yaml.safe_dump(config, sort_keys=False)
    config_path.write_text(config_text, encoding="utf-8")
    (result_dir / "config.yaml").write_text(config_text, encoding="utf-8")
    (result_dir / "command.txt").write_text(
        f"scripts/train_peft.py --config configs/{run_id}.yaml\n", encoding="utf-8"
    )
    (result_dir / "best.pt").write_bytes(f"best-{run_id}".encode())
    (result_dir / "last.pt").write_bytes(f"last-{run_id}".encode())
    history = [
        {
            "epoch": 1,
            "validation": {"all": {"top1": top1 - 0.001, "macro_accuracy": top1 - 0.002}},
        },
        {
            "epoch": 2,
            "validation": {"all": {"top1": top1, "macro_accuracy": top1 - 0.001}},
        },
    ]
    metrics = {
        "run_id": run_id,
        "config_sha256": _sha256(config_path),
        "manifest_sha256": MANIFEST_SHA,
        "split_sha256": SPLIT_SHA,
        "git_commit": COMMIT,
        "train_count": 80,
        "validation_count": 20,
        "scope": "train/validation PEFT; test data not accessed",
        "history": history,
        "best_top1": top1,
        "duration_seconds": 12.0,
        "peak_vram_mib": 64.0,
        "best_checkpoint_sha256": _sha256(result_dir / "best.pt"),
        "last_checkpoint_sha256": _sha256(result_dir / "last.pt"),
    }
    (result_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return {
        "run_id": run_id,
        "pair_id": pair_id,
        "role": role,
        "seed": seed,
        "expected_git_commit": COMMIT,
        "config": str(config_path.relative_to(root)),
        "result_dir": str(result_dir.relative_to(root)),
    }


def _make_plan(root: Path, runs: list[dict]) -> Path:
    plan_path = root / "plan.yaml"
    plan = {
        "batch": {
            "name": "fixture",
            "manifest_sha256": MANIFEST_SHA,
            "split_sha256": SPLIT_SHA,
            "train_count": 80,
            "validation_count": 20,
            "h1_reference_top1": 0.69,
            "minimum_m1_mean_gain": 0.001,
            "runs": runs,
        }
    }
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    return plan_path


def _complete_fixture(root: Path) -> tuple[Path, list[dict]]:
    runs = [
        _make_run(root, run_id="m0-s1", pair_id="s1", role="m0", seed=1, top1=0.680),
        _make_run(root, run_id="m1-s1", pair_id="s1", role="m1", seed=1, top1=0.682),
        _make_run(root, run_id="m0-s2", pair_id="s2", role="m0", seed=2, top1=0.681),
        _make_run(root, run_id="m1-s2", pair_id="s2", role="m1", seed=2, top1=0.683),
    ]
    return _make_plan(root, runs), runs


def test_complete_batch_is_verified_aggregated_and_gated(tmp_path: Path) -> None:
    plan, _ = _complete_fixture(tmp_path)
    readiness = check_readiness(plan, tmp_path)
    assert readiness["ready"] is True
    assert len(readiness["ready_runs"]) == 4

    report = aggregate_batch(plan, tmp_path)
    assert report["summary"]["m0"]["mean_top1"] == pytest.approx(0.6805)
    assert report["summary"]["m1"]["mean_top1"] == pytest.approx(0.6825)
    assert report["decision_gate"]["mean_paired_gain"] == pytest.approx(0.002)
    assert report["decision_gate"]["m1_stable_over_m0"] is True
    assert report["decision_gate"]["decision"] == (
        "retain_m1_mechanism_only_h1_remains_stronger"
    )
    assert all(run["best_epoch"] == 2 for run in report["runs"])


def test_only_line_ending_differences_are_accepted(tmp_path: Path) -> None:
    plan, runs = _complete_fixture(tmp_path)
    result_dir = tmp_path / runs[0]["result_dir"]
    returned_config = result_dir / "config.yaml"
    lf_text = returned_config.read_text(encoding="utf-8")
    returned_config.write_bytes(lf_text.replace("\n", "\r\n").encode("utf-8"))
    metrics_path = result_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["config_sha256"] = _sha256(returned_config)
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    report = aggregate_batch(plan, tmp_path)
    first = next(run for run in report["runs"] if run["run_id"] == runs[0]["run_id"])
    assert first["config_sha256"] == _sha256(returned_config)
    assert len(first["canonical_config_sha256"]) == 64


def test_checkpoint_tampering_is_rejected(tmp_path: Path) -> None:
    plan, runs = _complete_fixture(tmp_path)
    target = tmp_path / runs[0]["result_dir"] / "best.pt"
    target.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="best_checkpoint_sha256"):
        aggregate_batch(plan, tmp_path)


def test_readiness_reports_missing_worker_files(tmp_path: Path) -> None:
    plan, runs = _complete_fixture(tmp_path)
    target = tmp_path / runs[-1]["result_dir"] / "metrics.json"
    target.unlink()
    readiness = check_readiness(plan, tmp_path)
    assert readiness["ready"] is False
    assert str(target) in readiness["missing_files"]
