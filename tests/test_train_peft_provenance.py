from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_peft", ROOT / "scripts" / "train_peft.py")
assert SPEC is not None and SPEC.loader is not None
TRAIN_PEFT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN_PEFT)


def test_git_commit_uses_explicit_archive_commit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    expected = "1d8249b86d46594bdf30c6828d7dabb889c3a4e4"
    monkeypatch.setenv("AICTRACK1_GIT_COMMIT", expected)
    monkeypatch.setattr(
        TRAIN_PEFT.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(128, "git")),
    )
    assert TRAIN_PEFT._git_commit(tmp_path) == expected


def test_git_commit_marks_unverifiable_archive_unknown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("AICTRACK1_GIT_COMMIT", raising=False)
    monkeypatch.setattr(
        TRAIN_PEFT.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(128, "git")),
    )
    assert TRAIN_PEFT._git_commit(tmp_path) == "unknown_archive"
