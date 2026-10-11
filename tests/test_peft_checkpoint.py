from __future__ import annotations

import pytest
import torch
from torch import nn

from aictrack1.engine.peft import (
    configure_classifier_trainability_from_state,
    load_trainable_state_dict,
    trainable_state_dict,
)


def test_trainable_checkpoint_round_trip_excludes_frozen_parameters() -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    for parameter in model[0].parameters():
        parameter.requires_grad_(False)
    state = trainable_state_dict(model)
    assert set(state) == {"1.weight", "1.bias"}
    expected = {name: value.clone() for name, value in state.items()}
    with torch.no_grad():
        for parameter in model[1].parameters():
            parameter.zero_()
    load_trainable_state_dict(model, state)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert torch.equal(parameter, expected[name])


def test_trainable_checkpoint_rejects_key_mismatch() -> None:
    model = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="mismatch"):
        load_trainable_state_dict(model, {"weight": model.weight.detach().clone()})


@pytest.mark.parametrize("joint_head", [False, True])
def test_checkpoint_classifier_trainability_matches_frozen_or_joint_state(joint_head: bool) -> None:
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = nn.Linear(3, 3)
            self.classifier = nn.Linear(3, 2)

    source = TinyModel()
    for parameter in source.classifier.parameters():
        parameter.requires_grad_(joint_head)
    state = trainable_state_dict(source)
    restored = TinyModel()
    enabled = configure_classifier_trainability_from_state(restored.classifier, state)
    assert enabled is joint_head
    load_trainable_state_dict(restored, state)
    for name, parameter in restored.named_parameters():
        if name in state:
            assert torch.equal(parameter, state[name])


def test_checkpoint_classifier_rejects_partial_joint_state() -> None:
    classifier = nn.Linear(3, 2)
    with pytest.raises(ValueError, match="classifier checkpoint trainable parameter mismatch"):
        configure_classifier_trainability_from_state(
            classifier, {"classifier.weight": classifier.weight.detach().clone()}
        )
