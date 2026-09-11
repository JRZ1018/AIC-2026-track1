from __future__ import annotations

import pytest
import torch

from aictrack1.models.heads import (
    CosineClassifier,
    LinearClassifier,
    build_classifier_head,
    load_classifier_head_checkpoint,
)


def test_linear_and_cosine_heads_have_expected_shapes_and_gradients() -> None:
    features = torch.randn(7, 5, requires_grad=True)
    for head in (LinearClassifier(5, 3), CosineClassifier(5, 3)):
        logits = head(features)
        assert logits.shape == (7, 3)
        logits.sum().backward(retain_graph=True)
        assert all(parameter.grad is not None for parameter in head.parameters())


def test_cosine_classifier_is_invariant_to_positive_feature_scaling() -> None:
    head = CosineClassifier(4, 2, initial_scale=3.0, learnable_scale=False)
    features = torch.randn(6, 4)
    assert torch.allclose(head(features), head(features * 11), atol=1e-6)


def test_head_factory_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unsupported classifier"):
        build_classifier_head("ensemble", 4, 2)


@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_classifier_head_checkpoint_loads_both_supported_kinds(tmp_path, kind: str) -> None:
    torch.manual_seed(13)
    head = build_classifier_head(kind, 4, 3)
    checkpoint = tmp_path / f"{kind}.pt"
    torch.save(
        {
            "head": {"kind": kind, "initial_scale": 10.0, "learnable_scale": True},
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
