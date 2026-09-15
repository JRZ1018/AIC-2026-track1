from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "experiment"
BASELINE = CONFIG_ROOT / "p3_stage1_h1init_peft_lora_last2_r32a64_3ep_seed1_b_planned.yaml"
FILES = sorted(CONFIG_ROOT.glob("p3_stage1_overnight_*_seed1_b.yaml"))
COMMON_FIELDS = (
    "training_scope",
    "dataset_config",
    "model_config",
    "manifest",
    "split",
    "label_conflicts",
    "head_checkpoint",
    "device",
    "expected_initial_top1",
    "initial_top1_tolerance",
)


def _experiment(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["experiment"]


def test_overnight_queue_is_complete_unique_and_test_free() -> None:
    assert len(FILES) == 15
    baseline = _experiment(BASELINE)
    run_ids: set[str] = set()
    outputs: set[str] = set()

    for path in FILES:
        candidate = _experiment(path)
        assert candidate["execution_enabled"] is True
        assert candidate["training"]["epochs"] == 8
        assert candidate["training"]["batch_size"] == 64
        assert candidate["training"]["workers"] == 4
        assert candidate["training"]["seed"] == 20260910
        assert candidate["training"]["head_learning_rate"] == 0.0
        assert candidate["training"]["weight_decay"] == 0.01
        assert candidate["training"]["gradient_clip_norm"] == 1.0
        assert "test" not in " ".join(str(value) for value in candidate.values()).lower()
        for field in COMMON_FIELDS:
            expected = (
                "fixed_split_validation"
                if field == "training_scope" and field not in baseline
                else baseline[field]
            )
            assert candidate[field] == expected
        assert Path(candidate["output_dir"]).name == candidate["run_id"]
        run_ids.add(candidate["run_id"])
        outputs.add(candidate["output_dir"])

    assert len(run_ids) == len(FILES)
    assert len(outputs) == len(FILES)


def test_overnight_queue_is_the_preregistered_bounded_grid() -> None:
    t2 = set()
    lora = set()
    for path in FILES:
        candidate = _experiment(path)
        peft = candidate["peft"]
        lr = candidate["training"]["visual_learning_rate"]
        if peft["mode"] == "last_n_blocks":
            t2.add((peft["lora_blocks"], lr))
        else:
            assert peft["mode"] == "lora_last_n_attention"
            lora.add(
                (
                    peft["lora_blocks"],
                    peft["lora_rank"],
                    float(peft["lora_alpha"]),
                    lr,
                )
            )

    assert t2 == {
        (blocks, lr)
        for blocks in (2, 3, 4)
        for lr in (1e-6, 2e-6, 3e-6)
    }
    assert lora == {
        (2, 16, 32.0, 1e-4),
        (2, 32, 64.0, 5e-5),
        (2, 32, 64.0, 1e-4),
        (2, 32, 64.0, 2e-4),
        (3, 32, 64.0, 1e-4),
        (4, 32, 64.0, 1e-4),
    }


def test_overnight_queue_contains_at_least_eight_hours_of_measured_work() -> None:
    # Conservative floor from A's measured leak-free T2 run: 931.877 s / 3 epochs.
    total_epochs = sum(_experiment(path)["training"]["epochs"] for path in FILES)
    conservative_seconds_per_epoch = 931.877 / 3
    assert total_epochs == 120
    assert total_epochs * conservative_seconds_per_epoch >= 8 * 60 * 60


def test_overnight_runner_lists_every_config_once() -> None:
    runner = (ROOT / "scripts" / "run_b_overnight_queue.ps1").read_text(encoding="utf-8")
    for path in FILES:
        assert runner.count(path.name) == 1
    assert runner.count("scripts\\train_peft.py") == 1
    assert "--resume" in runner
    assert "infer_and_build_submission" not in runner
    assert "test-manifest" not in runner
    assert "2>&1" not in runner
    assert "| Tee-Object" not in runner
    assert "Start-Process" in runner
    assert ".WaitForExit()" in runner
    assert ".Refresh()" in runner
    assert "$null -eq $exitCode" in runner
    assert "Remove-Item -LiteralPath $output" in runner
    assert "Get-ChildItem -LiteralPath $output -Force" in runner
    assert "-RedirectStandardOutput $stdoutLog" in runner
    assert "-RedirectStandardError $stderrLog" in runner
