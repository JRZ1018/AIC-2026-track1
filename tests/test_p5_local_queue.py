import copy
import importlib.util
from pathlib import Path

import pytest
import yaml
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p5_local_queue", ROOT / "scripts/run_stage2_p5_local_fold1_pair.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def fixture_pair():
    reports = [{"four_view": {"top1": .7, "macro_accuracy": .69, "tail_class_mean_recall": .68},
                "center": {"top1": .69}, "evaluated_count": 29737, "smoke_only": False, "four_view_resize": 384},
               {"four_view": {"top1": .703, "macro_accuracy": .692, "tail_class_mean_recall": .679},
                "center": {"top1": .692}, "evaluated_count": 29737, "smoke_only": False, "four_view_resize": 384}]
    metrics = [{"best_top1": .69 if i == 0 else .692,
                "history": [{"epoch": e, "confidence_weighting": {"next_effective_quality": {"decision": "L1"}},
                             "gradient_diagnostics": {"all_finite": True}} for e in range(1, 9)]} for i in range(2)]
    return reports, metrics, {"paired_near_duplicate_group_bootstrap": {"ci95_low": .001}}


def test_local_pair_is_approved_but_remote_pair_stays_disabled():
    items = queue.validate_pair()
    assert len(items) == 2
    for tag in ("control", "candidate"):
        remote = yaml.safe_load((ROOT / f"configs/experiment/stage2_p5_sce_fold1_3090_{tag}.yaml").read_text())["experiment"]
        local = yaml.safe_load((ROOT / f"configs/experiment/stage2_p5_sce_fold1_local_{tag}.yaml").read_text())["experiment"]
        assert remote["execution_enabled"] is False
        assert local["runtime"]["record_gradient_diagnostics"] is True
        for key in ("training", "peft", "confidence_weighting", "head_checkpoint", "split", "manifest", "base_loss"):
            assert local[key] == remote[key]


def test_passing_screen_does_not_authorize_full_or_next_fold():
    result = queue.pair_checks(*fixture_pair())
    assert result["passed"]
    assert result["full_training_authorized"] is False
    assert result["next_fold_authorized"] is False


def test_safe_numpy_rng_checkpoint_is_weights_only_loadable_and_exact(tmp_path):
    rng = np.random.RandomState(20260921)
    state = rng.get_state()
    safe = (state[0], state[1].tolist(), *state[2:])
    path = tmp_path / "rng.pt"
    torch.save({"numpy_rng_state": safe}, path)
    restored = torch.load(path, weights_only=True, map_location="cpu")
    other = np.random.RandomState()
    other.set_state(restored["numpy_rng_state"])
    assert np.array_equal(rng.randn(256), other.randn(256))


@pytest.mark.parametrize("failure", ["small_gain", "ci", "macro", "tail", "count", "epochs", "l2", "center", "gradient"])
def test_local_gate_fails_closed(failure):
    reports, metrics, audit = copy.deepcopy(fixture_pair())
    if failure == "small_gain": reports[1]["four_view"]["top1"] = .701
    if failure == "ci": audit["paired_near_duplicate_group_bootstrap"]["ci95_low"] = 0
    if failure == "macro": reports[1]["four_view"]["macro_accuracy"] = .69
    if failure == "tail": reports[1]["four_view"]["tail_class_mean_recall"] = .677
    if failure == "count": reports[1]["evaluated_count"] = 64
    if failure == "epochs": metrics[1]["history"].pop()
    if failure == "l2": metrics[1]["history"][2]["confidence_weighting"]["next_effective_quality"]["decision"] = "L2"
    if failure == "center": reports[1]["center"]["top1"] = .68
    if failure == "gradient": metrics[1]["history"][2].pop("gradient_diagnostics")
    assert not queue.pair_checks(reports, metrics, audit)["passed"]
