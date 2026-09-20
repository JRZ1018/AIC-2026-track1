from __future__ import annotations

import pytest
import torch

from aictrack1.engine.frozen_head import FrozenHeadConfig, FrozenHeadTrainer, FrozenLossConfig
from aictrack1.engine.provenance import CheckpointProvenance
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


def test_adam_constant_lr_and_disabled_gradient_clipping(monkeypatch) -> None:
    features, labels = _data()
    config = FrozenHeadConfig(
        epochs=2,
        batch_size=9,
        learning_rate=0.01,
        gradient_clip_norm=None,
        optimizer="adam",
        scheduler="constant",
        amp=False,
        seed=29,
    )
    trainer = _trainer(config)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("gradient clipping must be disabled when configured as null")

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fail_if_called)
    first_lr = trainer.optimizer.param_groups[0]["lr"]
    loss = trainer.train_epoch(features, labels)

    assert isinstance(trainer.optimizer, torch.optim.Adam)
    assert torch.isfinite(torch.tensor(loss))
    assert trainer.optimizer.param_groups[0]["lr"] == first_lr


@pytest.mark.parametrize("optimizer", ["sgd", "adam", "adamw"])
def test_supported_optimizers_validate(optimizer: str) -> None:
    FrozenHeadConfig(optimizer=optimizer).validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("optimizer", "rmsprop", "optimizer"),
        ("scheduler", "linear", "scheduler"),
        ("gradient_clip_norm", 0.0, "gradient_clip_norm"),
    ],
)
def test_invalid_training_controls_are_rejected(field: str, value, message: str) -> None:
    raw = FrozenHeadConfig().__dict__ | {field: value}
    with pytest.raises(ValueError, match=message):
        FrozenHeadConfig(**raw).validate()


def test_checkpoint_without_scheduler_field_remains_loadable() -> None:
    features, labels = _data()
    config = FrozenHeadConfig(epochs=1, batch_size=9, learning_rate=0.2, amp=False)
    trainer = _trainer(config)
    trainer.train_epoch(features, labels)
    legacy_state = trainer.state_dict()
    del legacy_state["config"]["scheduler"]

    restored = _trainer(config)
    restored.load_state_dict(legacy_state)
    for name, expected in trainer.model.state_dict().items():
        assert torch.equal(expected, restored.model.state_dict()[name])


def test_frozen_head_resume_rejects_cross_stage_checkpoint() -> None:
    config = FrozenHeadConfig(epochs=1, batch_size=9, learning_rate=0.2, amp=False)
    stage2 = CheckpointProvenance(
        competition_stage="stage2",
        class_mapping_sha256="1" * 64,
        train_manifest_sha256="2" * 64,
        split_sha256="3" * 64,
        clip_weight_sha256="4" * 64,
        num_classes=3,
    )
    trainer = FrozenHeadTrainer(
        LinearClassifier(6, 3),
        config,
        device="cpu",
        num_classes=3,
        provenance=stage2,
    )
    state = trainer.state_dict() | {"competition_stage": "stage1"}
    restored = FrozenHeadTrainer(
        LinearClassifier(6, 3),
        config,
        device="cpu",
        num_classes=3,
        provenance=stage2,
    )
    with pytest.raises(ValueError, match="competition_stage mismatch"):
        restored.load_state_dict(state)
