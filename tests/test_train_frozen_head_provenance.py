from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_frozen_head", ROOT / "scripts" / "train_frozen_head.py"
)
assert SPEC is not None and SPEC.loader is not None
TRAIN_FROZEN_HEAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN_FROZEN_HEAD)


def test_git_commit_uses_explicit_archive_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = "1d8249b86d46594bdf30c6828d7dabb889c3a4e4"
    monkeypatch.setenv("AICTRACK1_GIT_COMMIT", expected)
    monkeypatch.setattr(
        TRAIN_FROZEN_HEAD.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(128, "git")
        ),
    )
    assert TRAIN_FROZEN_HEAD._git_commit(tmp_path) == expected


def test_git_commit_rejects_unverifiable_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("AICTRACK1_GIT_COMMIT", raising=False)
    monkeypatch.setattr(
        TRAIN_FROZEN_HEAD.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(128, "git")
        ),
    )
    with pytest.raises(RuntimeError, match="AICTRACK1_GIT_COMMIT"):
        TRAIN_FROZEN_HEAD._git_commit(tmp_path)
