from __future__ import annotations

import pytest
import torch

from aictrack1.engine.frozen_head import FrozenHeadConfig, FrozenHeadTrainer, FrozenLossConfig
from aictrack1.models.heads import LinearClassifier


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(44)
    labels = torch.arange(3).repeat_interleave(12)
    prototypes = torch.eye(3, 6)
    features = prototypes[labels] + 0.05 * torch.randn(36, 6, generator=generator)
    return features, labels


def _trainer(config: FrozenHeadConfig) -> FrozenHeadTrainer:
    torch.manual_seed(71)
    return FrozenHeadTrainer(LinearClassifier(6, 3), config, device="cpu", num_classes=3)


def test_frozen_head_training_and_epoch_boundary_resume_are_exact(tmp_path) -> None:
    features, labels = _data()
    config = FrozenHeadConfig(epochs=4, batch_size=9, learning_rate=0.2, amp=False, seed=13)
    uninterrupted = _trainer(config)
    for _ in range(4):
        uninterrupted.train_epoch(features, labels)

    interrupted = _trainer(config)
    for _ in range(2):
        interrupted.train_epoch(features, labels)
    checkpoint = tmp_path / "head.pt"
    interrupted.save_checkpoint(checkpoint)
    resumed = _trainer(config)
    resumed.load_checkpoint(checkpoint)
    for _ in range(2):
        resumed.train_epoch(features, labels)

    for name, expected in uninterrupted.model.state_dict().items():
        assert torch.equal(expected, resumed.model.state_dict()[name])
    metrics = resumed.evaluate(features, labels)
    assert metrics["top1"] > 0.95
    assert metrics["macro_accuracy"] > 0.95


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_frozen_head_cuda_amp_single_step_is_finite() -> None:
    features, labels = _data()
    config = FrozenHeadConfig(epochs=1, batch_size=12, learning_rate=0.05, amp=True, seed=19)
    torch.manual_seed(71)
    trainer = FrozenHeadTrainer(LinearClassifier(6, 3), config, device="cuda", num_classes=3)
    loss = trainer.train_epoch(features, labels)
    assert torch.isfinite(torch.tensor(loss))
    assert trainer.scaler.is_enabled()


@pytest.mark.parametrize(
    "loss_config",
    [
        FrozenLossConfig(kind="gce", gce_q=0.7),
        FrozenLossConfig(kind="logit_clip_cross_entropy", logit_clip_tau=1.5),
    ],
)
def test_frozen_head_robust_losses_train_and_resume(
    tmp_path, loss_config: FrozenLossConfig
) -> None:
    features, labels = _data()
    config = FrozenHeadConfig(epochs=2, batch_size=9, learning_rate=0.2, amp=False, seed=23)
    torch.manual_seed(71)
    trainer = FrozenHeadTrainer(
        LinearClassifier(6, 3),
        config,
        device="cpu",
        num_classes=3,
        loss_config=loss_config,
    )
    first_loss = trainer.train_epoch(features, labels)
    checkpoint = tmp_path / "robust.pt"
    trainer.save_checkpoint(checkpoint)
    torch.manual_seed(71)
    resumed = FrozenHeadTrainer(
        LinearClassifier(6, 3),
        config,
        device="cpu",
        num_classes=3,
        loss_config=loss_config,
    )
    resumed.load_checkpoint(checkpoint)
    second_loss = resumed.train_epoch(features, labels)
    assert torch.isfinite(torch.tensor(first_loss))
    assert torch.isfinite(torch.tensor(second_loss))


def test_resume_rejects_loss_configuration_mismatch(tmp_path) -> None:
    features, labels = _data()
    config = FrozenHeadConfig(epochs=1, batch_size=9, learning_rate=0.2, amp=False)
    trainer = FrozenHeadTrainer(
        LinearClassifier(6, 3), config, device="cpu", num_classes=3
    )
    trainer.train_epoch(features, labels)
    checkpoint = tmp_path / "ce.pt"
    trainer.save_checkpoint(checkpoint)
    mismatched = FrozenHeadTrainer(
        LinearClassifier(6, 3),
        config,
        device="cpu",
        num_classes=3,
        loss_config=FrozenLossConfig(kind="gce"),
    )
    with pytest.raises(ValueError, match="loss configuration"):
        mismatched.load_checkpoint(checkpoint)
