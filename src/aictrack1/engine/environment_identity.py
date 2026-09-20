"""Stable environment identities for reproducible experiment records."""

from __future__ import annotations

import hashlib
import json
import platform
import re
from pathlib import Path
from typing import Any, Mapping


ENVIRONMENT_SCHEMA = "aictrack1.environment.v1"
UNKNOWN_LEGACY = "unknown_legacy"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_environment_json(identity: Mapping[str, Any]) -> str:
    """Return the only byte representation used for environment hashing."""

    validate_environment_identity(identity)
    return json.dumps(identity, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def environment_hash(identity: Mapping[str, Any]) -> str:
    canonical = canonical_environment_json(identity).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_environment_identity(identity: Mapping[str, Any]) -> None:
    if identity.get("schema") != ENVIRONMENT_SCHEMA:
        raise ValueError(f"environment schema must be {ENVIRONMENT_SCHEMA!r}")
    required_paths = (
        ("python", "implementation"),
        ("python", "version"),
        ("platform", "system"),
        ("platform", "release"),
        ("platform", "version"),
        ("platform", "machine"),
        ("pytorch", "version"),
        ("dependencies", "requirements_lock_sha256"),
        ("model", "provider"),
        ("model", "name"),
        ("model", "clip_weight_sha256"),
    )
    for section, field in required_paths:
        nested = identity.get(section)
        if not isinstance(nested, Mapping) or nested.get(field) in (None, ""):
            raise ValueError(f"environment identity is missing {section}.{field}")
    required_present = (
        ("pytorch", "cuda_runtime"),
        ("pytorch", "cudnn"),
        ("accelerator", "device"),
        ("accelerator", "gpu"),
    )
    for section, field in required_present:
        nested = identity.get(section)
        if not isinstance(nested, Mapping) or field not in nested:
            raise ValueError(f"environment identity is missing {section}.{field}")
    if not identity["accelerator"]["device"]:
        raise ValueError("environment identity accelerator.device must not be empty")
    for section, field in (
        ("dependencies", "requirements_lock_sha256"),
        ("model", "clip_weight_sha256"),
    ):
        value = str(identity[section][field])
        if _SHA256.fullmatch(value) is None:
            raise ValueError(f"environment identity {section}.{field} must be SHA-256")


def capture_environment_identity(
    root: str | Path,
    *,
    torch_module: Any,
    device: Any,
    clip_weight_sha256: str,
) -> dict[str, Any]:
    """Capture a complete runtime identity without importing torch in this module."""

    root_path = Path(root).resolve()
    requirements_path = root_path / "requirements-lock.txt"
    if not requirements_path.is_file():
        raise FileNotFoundError(f"requirements lock is missing: {requirements_path}")
    device_text = str(device)
    gpu = None
    if getattr(device, "type", None) == "cuda":
        gpu = torch_module.cuda.get_device_name(device)
    cudnn_version = None
    backends = getattr(torch_module, "backends", None)
    cudnn = getattr(backends, "cudnn", None)
    if cudnn is not None:
        cudnn_version = cudnn.version()
    identity: dict[str, Any] = {
        "schema": ENVIRONMENT_SCHEMA,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "pytorch": {
            "version": str(torch_module.__version__),
            "cuda_runtime": getattr(torch_module.version, "cuda", None),
            "cudnn": cudnn_version,
        },
        "accelerator": {"device": device_text, "gpu": gpu},
        "dependencies": {
            "requirements_lock_path": "requirements-lock.txt",
            "requirements_lock_sha256": sha256_file(requirements_path),
        },
        "model": {
            "provider": "OpenAI",
            "name": "ViT-B/32",
            "clip_weight_sha256": clip_weight_sha256,
        },
    }
    validate_environment_identity(identity)
    return identity


def verified_environment_hash(payload: Mapping[str, Any]) -> str | None:
    """Return a verified hash from a modern metrics/environment payload, if present."""

    identity = payload.get("environment")
    claimed = payload.get("environment_hash")
    if not isinstance(identity, Mapping) or not isinstance(claimed, str):
        return None
    actual = environment_hash(identity)
    if claimed != actual:
        raise ValueError(f"environment hash mismatch: {claimed!r} != {actual!r}")
    return actual
