from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "transfer_workspace.py"


def load_transfer_module():
    spec = importlib.util.spec_from_file_location("transfer_workspace", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_create_verify_and_restore_round_trip(tmp_path: Path) -> None:
    module = load_transfer_module()
    source = tmp_path / "source"
    source.mkdir()
    payload = source / "private" / "state.bin"
    payload.parent.mkdir()
    payload.write_bytes(b"competition-private-state")

    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    module.ROOT = source
    module.REQUIRED_ENTRIES = (Path("private"),)
    module.CRITICAL_SHA256 = {"private/state.bin": digest}

    bundle = tmp_path / "bundle"
    module.create_bundle(bundle, include_features=False)
    manifest = module.verify_bundle(bundle)
    assert manifest["file_count"] == 1

    restored = tmp_path / "restored"
    restored.mkdir()
    module.ROOT = restored
    module.restore_bundle(bundle)
    assert (restored / "private" / "state.bin").read_bytes() == payload.read_bytes()


def test_dataset_is_opt_in() -> None:
    module = load_transfer_module()
    assert module.OPTIONAL_DATASET not in module.selected_entries(include_features=False)
    assert module.OPTIONAL_DATASET in module.selected_entries(
        include_features=False, include_dataset=True
    )


@pytest.mark.parametrize("raw", ["../outside", "a/../../outside"])
def test_manifest_rejects_parent_traversal(raw: str) -> None:
    module = load_transfer_module()
    with pytest.raises(ValueError, match="unsafe manifest path"):
        module.safe_relative(raw)
