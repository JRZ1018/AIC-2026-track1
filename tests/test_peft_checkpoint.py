from __future__ import annotations

import pytest
import torch
from torch import nn

from aictrack1.engine.peft import load_trainable_state_dict, trainable_state_dict


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
