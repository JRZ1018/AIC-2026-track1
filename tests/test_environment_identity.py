from __future__ import annotations

from types import SimpleNamespace

import pytest

from aictrack1.engine.environment_identity import (
    ENVIRONMENT_SCHEMA,
    capture_environment_identity,
    environment_hash,
    verified_environment_hash,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def _identity() -> dict[str, object]:
    return {
        "schema": ENVIRONMENT_SCHEMA,
        "python": {"implementation": "CPython", "version": "3.12.10"},
        "platform": {
            "system": "Windows",
            "release": "11",
            "version": "10.0.26100",
            "machine": "AMD64",
        },
        "pytorch": {"version": "2.11.0+cu128", "cuda_runtime": "12.8", "cudnn": 91002},
        "accelerator": {"device": "cuda", "gpu": "Synthetic GPU"},
        "dependencies": {
            "requirements_lock_path": "requirements-lock.txt",
            "requirements_lock_sha256": SHA_A,
        },
        "model": {
            "provider": "OpenAI",
            "name": "ViT-B/32",
            "clip_weight_sha256": SHA_B,
        },
    }


def test_environment_hash_is_canonical_and_change_sensitive() -> None:
    identity = _identity()
    reordered = {key: identity[key] for key in reversed(identity)}
    assert environment_hash(identity) == environment_hash(reordered)
    changed = _identity()
    changed["python"] = {"implementation": "CPython", "version": "3.12.11"}
    assert environment_hash(identity) != environment_hash(changed)


def test_environment_identity_requires_cuda_and_platform_keys() -> None:
    identity = _identity()
    identity["pytorch"] = {"version": "2.11.0", "cudnn": None}
    with pytest.raises(ValueError, match="pytorch.cuda_runtime"):
        environment_hash(identity)


def test_capture_environment_identity_covers_lock_model_and_runtime(tmp_path) -> None:
    (tmp_path / "requirements-lock.txt").write_text("torch==2.11.0\n", encoding="utf-8")

    class Device:
        type = "cuda"

        def __str__(self) -> str:
            return "cuda:0"

    fake_torch = SimpleNamespace(
        __version__="2.11.0+cu128",
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(get_device_name=lambda _device: "Synthetic GPU"),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 91002)),
    )
    identity = capture_environment_identity(
        tmp_path,
        torch_module=fake_torch,
        device=Device(),
        clip_weight_sha256=SHA_B,
    )
    assert identity["pytorch"]["cuda_runtime"] == "12.8"
    assert identity["accelerator"] == {"device": "cuda:0", "gpu": "Synthetic GPU"}
    assert identity["dependencies"]["requirements_lock_sha256"] != SHA_A
    assert identity["model"]["clip_weight_sha256"] == SHA_B
    digest = environment_hash(identity)
    assert verified_environment_hash({"environment": identity, "environment_hash": digest}) == digest


def test_cudnn_probe_failure_is_recorded_without_losing_the_run(tmp_path) -> None:
    (tmp_path / "requirements-lock.txt").write_text("torch==2.11.0\n", encoding="utf-8")

    class Device:
        type = "cuda"

        def __str__(self) -> str:
            return "cuda:0"

    def failed_probe() -> int:
        raise RuntimeError("compiled 9.19 but found 9.2")

    fake_torch = SimpleNamespace(
        __version__="2.11.0+cu128",
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(get_device_name=lambda _device: "Synthetic GPU"),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=failed_probe)),
    )
    identity = capture_environment_identity(
        tmp_path,
        torch_module=fake_torch,
        device=Device(),
        clip_weight_sha256=SHA_B,
    )
    assert identity["pytorch"]["cudnn"] is None
    assert "compiled 9.19 but found 9.2" in identity["pytorch"]["cudnn_probe_error"]
    assert environment_hash(identity)


def test_verified_environment_hash_rejects_tampering() -> None:
    with pytest.raises(ValueError, match="environment hash mismatch"):
        verified_environment_hash({"environment": _identity(), "environment_hash": SHA_A})
