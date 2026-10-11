from __future__ import annotations

import pytest
import torch

from aictrack1.engine.provenance import CheckpointProvenance
from aictrack1.models.heads import (
    CosineClassifier,
    FrozenFeatureImageClassifier,
    LinearClassifier,
    ResidualMLPClassifier,
    build_classifier_head,
    load_classifier_head_checkpoint,
    tau_normalize_linear_classifier,
)


def test_frozen_feature_image_classifier_normalizes_visual_features() -> None:
    visual = torch.nn.Linear(3, 4, bias=False)
    head = LinearClassifier(4, 2)
    model = FrozenFeatureImageClassifier(visual, head)
    images = torch.randn(5, 3)
    expected = head(torch.nn.functional.normalize(visual(images).float(), dim=-1))
    assert torch.equal(model(images), expected)


def test_supported_heads_have_expected_shapes_and_gradients() -> None:
    features = torch.randn(7, 5, requires_grad=True)
    for head in (
        LinearClassifier(5, 3),
        CosineClassifier(5, 3),
        ResidualMLPClassifier(5, 3, hidden_dim=4, dropout=0.0),
    ):
        logits = head(features)
        assert logits.shape == (7, 3)
        logits.sum().backward(retain_graph=True)
        assert all(parameter.grad is not None for parameter in head.parameters())


def test_residual_mlp_starts_as_its_linear_path() -> None:
    head = ResidualMLPClassifier(5, 3, hidden_dim=4, dropout=0.25)
    features = torch.randn(7, 5)
    head.train()
    assert torch.equal(head(features), head.linear(features))


@pytest.mark.parametrize(
    ("hidden_dim", "dropout"),
    [(0, 0.1), (4, -0.1), (4, 1.0)],
)
def test_residual_mlp_rejects_invalid_shape_or_dropout(
    hidden_dim: int, dropout: float
) -> None:
    with pytest.raises(ValueError, match="residual"):
        ResidualMLPClassifier(5, 3, hidden_dim=hidden_dim, dropout=dropout)


def test_cosine_classifier_is_invariant_to_positive_feature_scaling() -> None:
    head = CosineClassifier(4, 2, initial_scale=3.0, learnable_scale=False)
    features = torch.randn(6, 4)
    assert torch.allclose(head(features), head(features * 11), atol=1e-6)


def test_head_factory_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unsupported classifier"):
        build_classifier_head("ensemble", 4, 2)


def test_tau_normalization_is_train_free_and_does_not_mutate_source() -> None:
    head = LinearClassifier(2, 2)
    with torch.no_grad():
        head.linear.weight.copy_(torch.tensor([[3.0, 4.0], [0.0, 2.0]]))
        head.linear.bias.copy_(torch.tensor([1.0, -1.0]))
    calibrated = tau_normalize_linear_classifier(head, 1.0)
    assert torch.allclose(calibrated.linear.weight.norm(dim=1), torch.ones(2))
    assert torch.equal(calibrated.linear.bias, head.linear.bias)
    assert torch.equal(head.linear.weight, torch.tensor([[3.0, 4.0], [0.0, 2.0]]))
    assert torch.equal(tau_normalize_linear_classifier(head, 0.0).linear.weight, head.linear.weight)


@pytest.mark.parametrize("tau", [-0.1, 1.1, float("nan")])
def test_tau_normalization_rejects_invalid_values(tau: float) -> None:
    with pytest.raises(ValueError, match="tau must"):
        tau_normalize_linear_classifier(LinearClassifier(2, 2), tau)


@pytest.mark.parametrize("kind", ["linear", "cosine", "residual_mlp"])
def test_classifier_head_checkpoint_loads_supported_kinds(tmp_path, kind: str) -> None:
    torch.manual_seed(13)
    head = build_classifier_head(
        kind,
        4,
        3,
        residual_hidden_dim=5,
        residual_dropout=0.0,
    )
    checkpoint = tmp_path / f"{kind}.pt"
    torch.save(
        {
            "head": {
                "kind": kind,
                "initial_scale": 10.0,
                "learnable_scale": True,
                "hidden_dim": 5,
                "dropout": 0.0,
            },
            "feature_cache_metadata": {
                "clip_weight_sha256": "clip",
                "class_mapping_sha256": "mapping",
                "split_sha256": "split",
            },
            "model": head.state_dict(),
        },
        checkpoint,
    )
    restored, payload = load_classifier_head_checkpoint(
        checkpoint,
        feature_dim=4,
        num_classes=3,
        expected_clip_weight_sha256="clip",
        expected_class_mapping_sha256="mapping",
        expected_split_sha256="split",
    )
    features = torch.randn(2, 4)
    assert payload["head"]["kind"] == kind
    assert torch.equal(head(features), restored(features))


def test_classifier_head_checkpoint_rejects_provenance_mismatch(tmp_path) -> None:
    head = LinearClassifier(4, 3)
    checkpoint = tmp_path / "head.pt"
    torch.save(
        {
            "head": {"kind": "linear"},
            "feature_cache_metadata": {
                "clip_weight_sha256": "wrong",
                "class_mapping_sha256": "mapping",
                "split_sha256": "split",
            },
            "model": head.state_dict(),
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="clip_weight_sha256 mismatch"):
        load_classifier_head_checkpoint(
            checkpoint,
            feature_dim=4,
            num_classes=3,
            expected_clip_weight_sha256="clip",
            expected_class_mapping_sha256="mapping",
            expected_split_sha256="split",
        )


def test_classifier_head_checkpoint_rejects_full_train_initializer_for_validation(
    tmp_path,
) -> None:
    head = LinearClassifier(4, 3)
    checkpoint = tmp_path / "full-head.pt"
    torch.save(
        {
            "head": {"kind": "linear"},
            "feature_cache_metadata": {
                "clip_weight_sha256": "clip",
                "class_mapping_sha256": "mapping",
                "split_sha256": "split",
                "training_scope": "full_train_fixed_epochs",
                "training_record_count": 103218,
            },
            "model": head.state_dict(),
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="cannot initialize a fixed-split validation run"):
        load_classifier_head_checkpoint(
            checkpoint,
            feature_dim=4,
            num_classes=3,
            expected_clip_weight_sha256="clip",
            expected_class_mapping_sha256="mapping",
            expected_split_sha256="split",
            forbid_full_train_initializer=True,
        )


def test_classifier_head_checkpoint_rejects_cross_stage_provenance(tmp_path) -> None:
    expected = CheckpointProvenance(
        competition_stage="stage2",
        class_mapping_sha256="1" * 64,
        train_manifest_sha256="2" * 64,
        split_sha256="3" * 64,
        clip_weight_sha256="4" * 64,
        num_classes=3,
    )
    head = LinearClassifier(4, 3)
    checkpoint = tmp_path / "stage1-head.pt"
    torch.save(
        {
            **expected.to_dict(),
            "competition_stage": "stage1",
            "head": {"kind": "linear"},
            "feature_cache_metadata": {
                "clip_weight_sha256": expected.clip_weight_sha256,
                "class_mapping_sha256": expected.class_mapping_sha256,
                "split_sha256": expected.split_sha256,
            },
            "model": head.state_dict(),
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="competition_stage mismatch"):
        load_classifier_head_checkpoint(
            checkpoint,
            feature_dim=4,
            num_classes=3,
            expected_clip_weight_sha256=expected.clip_weight_sha256,
            expected_class_mapping_sha256=expected.class_mapping_sha256,
            expected_split_sha256=expected.split_sha256,
            expected_provenance=expected,
        )
