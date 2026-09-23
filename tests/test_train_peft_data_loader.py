from __future__ import annotations

import pytest
import torch
from torch.utils.data import TensorDataset

from scripts.train_peft import _data_loader_settings, _loader, _runtime_settings


def test_data_loader_settings_use_backward_compatible_defaults() -> None:
    assert _data_loader_settings({}) == (0, 2, True)
    assert _data_loader_settings({"workers": 8}) == (8, 2, True)


def test_data_loader_settings_validate_runtime_fields() -> None:
    assert _data_loader_settings(
        {"workers": 12, "prefetch_factor": 4, "persistent_workers": False}
    ) == (12, 4, False)
    with pytest.raises(ValueError, match="prefetch_factor"):
        _data_loader_settings({"workers": 4, "prefetch_factor": 0})
    with pytest.raises(ValueError, match="persistent_workers"):
        _data_loader_settings({"workers": 4, "persistent_workers": "false"})


def test_runtime_settings_default_to_legacy_path_and_validate_opt_in() -> None:
    assert _runtime_settings({}) == {
        "defer_epoch_metrics": False,
        "reuse_validation_loader": False,
    }
    assert _runtime_settings(
        {"runtime": {"defer_epoch_metrics": True, "reuse_validation_loader": True}}
    ) == {
        "defer_epoch_metrics": True,
        "reuse_validation_loader": True,
    }
    with pytest.raises(ValueError, match="defer_epoch_metrics"):
        _runtime_settings({"runtime": {"defer_epoch_metrics": "true"}})
    with pytest.raises(ValueError, match="reuse_validation_loader"):
        _runtime_settings({"runtime": {"reuse_validation_loader": "true"}})


def test_loader_applies_prefetch_only_with_worker_processes() -> None:
    dataset = TensorDataset(torch.arange(8))
    multiprocessing_loader = _loader(
        dataset,
        batch_size=2,
        workers=2,
        prefetch_factor=4,
        persistent_workers=False,
        shuffle=False,
        seed=7,
        device=torch.device("cpu"),
    )
    assert multiprocessing_loader.num_workers == 2
    assert multiprocessing_loader.prefetch_factor == 4
    assert multiprocessing_loader.persistent_workers is False

    synchronous_loader = _loader(
        dataset,
        batch_size=2,
        workers=0,
        prefetch_factor=8,
        persistent_workers=True,
        shuffle=False,
        seed=7,
        device=torch.device("cpu"),
    )
    assert synchronous_loader.num_workers == 0
    assert synchronous_loader.prefetch_factor is None
    assert synchronous_loader.persistent_workers is False
