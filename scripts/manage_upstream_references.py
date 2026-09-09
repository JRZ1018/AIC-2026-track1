"""Synchronize read-only upstream source snapshots at audited commit SHAs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "upstream_registry.yaml"


def run_git(args: list[str], cwd: Path, *, capture: bool = False) -> str:
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=capture,
    )
    return proc.stdout.strip() if capture else ""


def load_registry() -> tuple[Path, list[dict[str, object]]]:
    data = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    local_root = ROOT / str(data["policy"]["local_root"])
    local_root = local_root.resolve()
    expected_root = (ROOT / "references" / "upstream").resolve()
    if local_root != expected_root:
        raise RuntimeError(f"refusing unexpected upstream root: {local_root}")
    return local_root, data["repositories"]


def ensure_clean(repo_dir: Path) -> None:
    dirty = run_git(["status", "--porcelain"], repo_dir, capture=True)
    if dirty:
        raise RuntimeError(f"refusing to overwrite modified reference: {repo_dir}")


def sync_one(base: Path, item: dict[str, object]) -> None:
    repo_id = str(item["id"])
    repo_dir = (base / repo_id).resolve()
    if repo_dir.parent != base:
        raise RuntimeError(f"unsafe repository id: {repo_id}")
    url = str(item["url"]) + ".git"
    commit = str(item["commit"])

    if repo_dir.exists():
        if not (repo_dir / ".git").is_dir():
            raise RuntimeError(f"target exists but is not a Git repository: {repo_dir}")
        ensure_clean(repo_dir)
        current_url = run_git(["remote", "get-url", "origin"], repo_dir, capture=True)
        if current_url.rstrip("/") != url.rstrip("/"):
            raise RuntimeError(f"origin mismatch for {repo_id}: {current_url}")
        try:
            current_head = run_git(["rev-parse", "--verify", "HEAD"], repo_dir, capture=True)
        except subprocess.CalledProcessError:
            current_head = ""
        if current_head == commit:
            print(f"keep {repo_id} @ {commit[:12]}")
            return
    else:
        repo_dir.mkdir(parents=False)
        run_git(["init", "--quiet"], repo_dir)
        run_git(["remote", "add", "origin", url], repo_dir)

    print(f"sync {repo_id} @ {commit[:12]}")
    run_git(["-c", "protocol.version=2", "fetch", "--quiet", "--depth=1", "--no-tags", "--filter=blob:none", "origin", commit], repo_dir)
    run_git(["checkout", "--quiet", "--detach", "FETCH_HEAD"], repo_dir)
    actual = run_git(["rev-parse", "HEAD"], repo_dir, capture=True)
    if actual != commit:
        raise RuntimeError(f"HEAD mismatch for {repo_id}: expected {commit}, got {actual}")


def check_one(base: Path, item: dict[str, object]) -> None:
    repo_id = str(item["id"])
    repo_dir = (base / repo_id).resolve()
    if not (repo_dir / ".git").is_dir():
        raise RuntimeError(f"missing reference repository: {repo_id}")
    expected_url = str(item["url"]) + ".git"
    expected_commit = str(item["commit"])
    actual_url = run_git(["remote", "get-url", "origin"], repo_dir, capture=True)
    actual_commit = run_git(["rev-parse", "HEAD"], repo_dir, capture=True)
    if actual_url.rstrip("/") != expected_url.rstrip("/"):
        raise RuntimeError(f"origin mismatch for {repo_id}: {actual_url}")
    if actual_commit != expected_commit:
        raise RuntimeError(f"HEAD mismatch for {repo_id}: {actual_commit}")
    ensure_clean(repo_dir)
    print(f"ok   {repo_id} @ {actual_commit[:12]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("sync", "check"))
    args = parser.parse_args()
    try:
        base, repositories = load_registry()
        base.mkdir(parents=True, exist_ok=True)
        for item in repositories:
            (sync_one if args.action == "sync" else check_one)(base, item)
    except (KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"upstream {args.action} FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"upstream {args.action} PASSED: {len(repositories)} repositories")
    return 0


if __name__ == "__main__":
    sys.exit(main())
