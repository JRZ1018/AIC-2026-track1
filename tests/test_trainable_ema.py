import sys
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aictrack1.engine.ema import TrainableEMA, optimizer_step_completed


def model() -> nn.Module:
    net = nn.Sequential(nn.Linear(2, 1, bias=False), nn.Linear(1, 1, bias=False))
    net[1].weight.requires_grad_(False)
    return net


def test_ema_tracks_only_trainable_parameters():
    net = model()
    ema = TrainableEMA(net, decay=0.5)
    assert set(ema.shadow) == {"0.weight"}
    initial = net[0].weight.detach().clone()
    with torch.no_grad():
        net[0].weight.add_(2)
    ema.update()
    assert ema.updates == 1
    assert torch.allclose(ema.shadow["0.weight"], initial + 1)


def test_ema_applied_restores_after_failure():
    net = model()
    ema = TrainableEMA(net, decay=0.5)
    with torch.no_grad():
        net[0].weight.add_(2)
    raw = net[0].weight.detach().clone()
    with pytest.raises(RuntimeError, match="sentinel"):
        with ema.applied():
            assert torch.allclose(net[0].weight, raw - 2)
            raise RuntimeError("sentinel")
    assert torch.equal(net[0].weight, raw)


def test_ema_weights_only_roundtrip(tmp_path):
    net = model()
    ema = TrainableEMA(net, decay=0.999)
    ema.update()
    path = tmp_path / "ema.pt"
    torch.save(ema.state_dict(), path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    restored = TrainableEMA(model(), decay=0.999)
    restored.load_state_dict(state)
    assert restored.updates == 1
    assert torch.equal(restored.shadow["0.weight"], ema.shadow["0.weight"])


def test_ema_rejects_wrong_decay_and_names():
    ema = TrainableEMA(model(), decay=0.999)
    state = ema.state_dict()
    state["decay"] = 0.99
    with pytest.raises(ValueError, match="decay"):
        ema.load_state_dict(state)
    state["decay"] = 0.999
    state["shadow"] = {}
    with pytest.raises(ValueError, match="names"):
        ema.load_state_dict(state)


def test_scaler_overflow_skips_ema_update():
    assert not optimizer_step_completed(scaler_enabled=True, before=65536, after=32768)
    assert optimizer_step_completed(scaler_enabled=True, before=32768, after=32768)
    assert optimizer_step_completed(scaler_enabled=True, before=32768, after=65536)
    assert optimizer_step_completed(scaler_enabled=False, before=1, after=1)


def test_p17_config_changes_only_ema_and_identity():
    root = Path(__file__).resolve().parents[1]
    control = yaml.safe_load((root / "configs/experiment/stage2_p13_fold2_sce_ac_8ep_4090_control.yaml").read_text(encoding="utf-8"))["experiment"]
    candidate = yaml.safe_load((root / "configs/experiment/stage2_p17_fold2_sce_ac_ema0999_8ep_4090.yaml").read_text(encoding="utf-8"))["experiment"]
    assert candidate.pop("trainable_ema") == {"enabled": True, "decay": 0.999}
    # Worker count is a user-approved DataLoader throughput setting, not an EMA algorithm change.
    assert control["training"]["workers"] == 4
    assert candidate["training"]["workers"] == 16
    control["training"]["workers"] = candidate["training"]["workers"]
    for name in ("run_id", "output_dir", "execution_enabled"):
        control.pop(name)
        candidate.pop(name)
    assert candidate == control
