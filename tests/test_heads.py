from __future__ import annotations

import pytest
import torch

from aictrack1.models.heads import CosineClassifier, LinearClassifier, build_classifier_head


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
