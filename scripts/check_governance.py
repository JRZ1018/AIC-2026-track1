"""Validate project governance metadata and Git safety without modifying files."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.contracts import validate_competition_stage_references


REGISTRY = ROOT / "docs" / "upstream_registry.yaml"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_FILES = (
    ROOT / "AGENTS.md",
    ROOT / ".gitignore",
    ROOT / "docs" / "HANDOFF.md",
    ROOT / "docs" / "00_获奖导向总体工作计划.md",
    ROOT / "docs" / "04_上游方案调研与复用登记.md",
    ROOT / "docs" / "05_上游最小复用规格.md",
    ROOT / "docs" / "compliance_questions.md",
    ROOT / "experiments" / "index.csv",
    ROOT / "submissions" / "ledger.csv",
    ROOT / "references" / "README.md",
    ROOT / "references" / "upstream" / ".gitkeep",
    ROOT / "scripts" / "manage_upstream_references.py",
)
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".pem", ".key"}
MAX_GIT_FILE_BYTES = 50 * 1024 * 1024


def git_candidates() -> list[Path]:
    proc = subprocess.run(
        ["git", "-c", "core.quotepath=false", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return [ROOT / line for line in proc.stdout.splitlines() if line]


def validate_registry(errors: list[str]) -> None:
    data = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    repos = data.get("repositories", [])
    allowed_decisions = set(data.get("policy", {}).get("allowed_decisions", []))
    if data.get("policy", {}).get("local_root") != "references/upstream":
        errors.append("upstream local_root must remain references/upstream")
    if not repos:
        errors.append("upstream registry has no repositories")
        return

    seen: set[str] = set()
    audit_text = (ROOT / "docs" / "04_上游方案调研与复用登记.md").read_text(encoding="utf-8")
    for item in repos:
        repo_id = item.get("id", "<missing-id>")
        if repo_id in seen:
            errors.append(f"duplicate upstream id: {repo_id}")
        seen.add(repo_id)
        for field in ("repository", "url", "default_branch", "commit", "license", "tier", "trigger_stages", "deliverable", "copy_policy", "decision"):
            if not item.get(field):
                errors.append(f"{repo_id}: missing {field}")
        commit = str(item.get("commit", ""))
        if not SHA_RE.fullmatch(commit):
            errors.append(f"{repo_id}: commit is not a full lowercase SHA")
        elif commit not in audit_text:
            errors.append(f"{repo_id}: locked commit is absent from human-readable audit")
        no_license = item.get("license") == "NO-LICENSE-FOUND"
        if no_license and item.get("copy_policy") != "forbidden":
            errors.append(f"{repo_id}: missing license must imply forbidden copy policy")
        if item.get("decision") not in allowed_decisions:
            errors.append(f"{repo_id}: unsupported decision {item.get('decision')!r}")
        if not isinstance(item.get("trigger_stages"), list):
            errors.append(f"{repo_id}: trigger_stages must be a list")


def validate_git_safety(errors: list[str]) -> None:
    for path in git_candidates():
        rel = path.relative_to(ROOT).as_posix()
        rel_lower = rel.lower()
        if rel_lower.startswith("data/") and rel_lower != "data/.gitkeep":
            errors.append(f"official/derived data is visible to Git: {rel}")
        if rel_lower.startswith("artifacts/"):
            errors.append(f"artifact is visible to Git: {rel}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"weight or secret-like file is visible to Git: {rel}")
        if re.search(r"(^|/)(credentials|service-account|auth-token|access-token)[^/]*\.json$", rel_lower):
            errors.append(f"credential-like file is visible to Git: {rel}")
        if path.is_file() and path.stat().st_size > MAX_GIT_FILE_BYTES:
            errors.append(f"file larger than 50 MiB is visible to Git: {rel}")


def validate_stage_isolation(errors: list[str]) -> None:
    """Validate every explicitly stage-scoped YAML without resolving private assets."""

    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        rel = path.relative_to(ROOT).as_posix()
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            errors.append(f"cannot parse configuration {rel}: {exc}")
            continue
        if not isinstance(document, dict):
            continue
        payload = document.get("experiment", document.get("dataset", document))
        if not isinstance(payload, dict):
            continue
        competition_stage = payload.get("competition_stage")
        is_stage2_named = "stage2" in path.name.lower()
        if is_stage2_named and competition_stage != "stage2":
            errors.append(f"Stage 2 configuration must declare competition_stage: stage2: {rel}")
            continue
        if competition_stage is None:
            continue
        try:
            validate_competition_stage_references(
                str(competition_stage), payload, context=rel
            )
        except ValueError as exc:
            errors.append(str(exc))


def main() -> int:
    errors: list[str] = []
    for path in REQUIRED_FILES:
        if not path.is_file():
            errors.append(f"required file missing: {path.relative_to(ROOT)}")
    if REGISTRY.is_file():
        validate_registry(errors)
    validate_stage_isolation(errors)
    if (ROOT / ".git").exists():
        try:
            validate_git_safety(errors)
        except subprocess.CalledProcessError as exc:
            errors.append(f"git safety check failed: {exc}")
    else:
        errors.append("Git repository is not initialized")

    if errors:
        print("Governance check FAILED:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Governance check PASSED")
    print(f"- upstream entries: {len(yaml.safe_load(REGISTRY.read_text(encoding='utf-8'))['repositories'])}")
    print("- explicitly stage-scoped configurations contain no cross-stage references")
    print("- Git candidate files contain no forbidden data, weights, secrets, or >50 MiB files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
