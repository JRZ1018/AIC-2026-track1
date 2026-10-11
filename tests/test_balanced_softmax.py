from __future__ import annotations

import io
import pytest
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path

from aictrack1.engine.frozen_head import FrozenHeadConfig, FrozenHeadTrainer, FrozenLossConfig
from aictrack1.methods.balanced_softmax import balanced_softmax_cross_entropy
from aictrack1.models.heads import LinearClassifier
from scripts.aggregate_stage2_p19_head_pair import _combined_group_bootstrap


def test_equal_training_counts_reduce_to_ordinary_ce() -> None:
    logits = torch.tensor([[1.2, -0.4, 0.1], [0.2, 1.0, -0.7]], requires_grad=True)
    targets = torch.tensor([0, 1])
    loss = balanced_softmax_cross_entropy(logits, targets, torch.full((3,), 12.0))
    assert torch.allclose(loss, F.cross_entropy(logits, targets), atol=1e-7)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_unequal_counts_apply_exact_log_support_offset() -> None:
    logits = torch.tensor([[0.4, -0.2], [-0.1, 0.7]], requires_grad=True)
    targets = torch.tensor([0, 1])
    counts = torch.tensor([3.0, 30.0])
    actual = balanced_softmax_cross_entropy(logits, targets, counts)
    expected = F.cross_entropy(logits + counts.log(), targets)
    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("counts", [torch.tensor([0.0, 1.0]), torch.tensor([1.0, float("nan")])])
def test_invalid_training_support_is_rejected(counts: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        balanced_softmax_cross_entropy(torch.zeros(1, 2), torch.zeros(1, dtype=torch.long), counts)


def test_resume_requires_the_exact_same_train_partition_counts() -> None:
    config = FrozenHeadConfig(epochs=1, batch_size=2, learning_rate=0.01, amp=False)
    loss = FrozenLossConfig(kind="balanced_softmax")
    torch.manual_seed(9)
    trainer = FrozenHeadTrainer(
        LinearClassifier(4, 2), config, device="cpu", num_classes=2,
        loss_config=loss, class_counts=torch.tensor([3, 6]),
    )
    state = trainer.state_dict()
    torch.manual_seed(9)
    identical = FrozenHeadTrainer(
        LinearClassifier(4, 2), config, device="cpu", num_classes=2,
        loss_config=loss, class_counts=torch.tensor([3, 6]),
    )
    identical.load_state_dict(state)
    assert identical.epoch == 0
    torch.manual_seed(9)
    mismatched = FrozenHeadTrainer(
        LinearClassifier(4, 2), config, device="cpu", num_classes=2,
        loss_config=loss, class_counts=torch.tensor([4, 5]),
    )
    with pytest.raises(ValueError, match="class counts mismatch"):
        mismatched.load_state_dict(state)


def test_new_last_state_can_be_loaded_with_weights_only() -> None:
    config = FrozenHeadConfig(epochs=2, batch_size=3, learning_rate=0.01, amp=False)
    loss = FrozenLossConfig(kind="balanced_softmax")
    counts = torch.tensor([3, 6])
    torch.manual_seed(9)
    trainer = FrozenHeadTrainer(
        LinearClassifier(4, 2), config, device="cpu", num_classes=2,
        loss_config=loss, class_counts=counts,
    )
    trainer.train_epoch(torch.randn(9, 4), torch.tensor([0] * 3 + [1] * 6))
    buffer = io.BytesIO()
    torch.save(trainer.state_dict(), buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location="cpu", weights_only=True)
    torch.manual_seed(9)
    restored = FrozenHeadTrainer(
        LinearClassifier(4, 2), config, device="cpu", num_classes=2,
        loss_config=loss, class_counts=counts,
    )
    restored.load_state_dict(loaded)
    assert restored.epoch == 1
    assert restored.optimizer.state_dict()["state"].keys() == trainer.optimizer.state_dict()["state"].keys()


def test_p19_combined_bootstrap_resamples_shared_groups_once() -> None:
    grouped = {
        "near:shared": [4, 1, 3],
        "singleton:a": [1, 0, 1],
        "singleton:b": [1, 1, 0],
    }
    first = _combined_group_bootstrap(grouped)
    second = _combined_group_bootstrap(grouped)
    assert first == second
    assert first["group_count"] == 3
    assert first["observation_count"] == 6
    assert first["delta"] == pytest.approx(2 / 6)


@pytest.mark.parametrize("fold", [2, 3])
def test_p19_config_changes_only_the_preregistered_loss(fold: int) -> None:
    root = Path(__file__).resolve().parents[1]
    config_root = root / "configs" / "experiment"
    baseline = yaml.safe_load(
        (config_root / f"stage2_fold{fold}_c5_h1_336_seed20260921.yaml").read_text(encoding="utf-8")
    )["experiment"]
    candidate = yaml.safe_load(
        (config_root / f"stage2_p19_fold{fold}_h1_balanced_softmax_336_seed20260921.yaml").read_text(encoding="utf-8")
    )["experiment"]
    assert candidate["execution_enabled"] is True  # Explicit P19 batch approval on 2026-10-01.
    assert candidate["loss"] == {"kind": "balanced_softmax"}
    for key in ("run_id", "output_dir", "execution_enabled", "loss"):
        baseline.pop(key, None)
        candidate.pop(key, None)
    assert candidate == baseline
