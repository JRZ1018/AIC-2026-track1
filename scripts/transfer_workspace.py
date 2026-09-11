"""Create, verify, and restore a private USB bundle for this workspace.

Only Git-ignored competition data and local artifacts are copied. Source code
is restored from GitHub.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "transfer_manifest.json"

REQUIRED_ENTRIES = (
    Path("artifacts/weights/ViT-B-32.pt"),
    Path("artifacts/audits/stage1"),
    Path("artifacts/experiments/20260910-p3-a1-cosine-ce"),
    Path("artifacts/experiments/20260910-p3-m1-classwise-history-3ep"),
    Path("submissions/20260911_m1_first"),
)
OPTIONAL_DATASET = Path("DataSetStage1")
OPTIONAL_FEATURE_CACHE = Path("artifacts/features/stage1/train_openai_vit_b32")

CRITICAL_SHA256 = {
    "artifacts/weights/ViT-B-32.pt": (
        "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
    ),
    "artifacts/experiments/20260910-p3-a1-cosine-ce/best.pt": (
        "6dbdc41c37b3422800aa5f11c30a7b70422849cd6aa62ce86803928c06dbec55"
    ),
    "artifacts/experiments/20260910-p3-m1-classwise-history-3ep/best.pt": (
        "4b489a872620e0be46e0b0bf5798cb5f6055a89ccbe13363c66884739b896f18"
    ),
    "submissions/20260911_m1_first/pred_results.csv": (
        "bcd3db88d044583b90628e8dfd0a7ff4ab1ec43ad00a0306e18bbf30a389c8ea"
    ),
    "submissions/20260911_m1_first/pred_results.zip": (
        "8acd23de9590eaab4018dbc085c5c5ce35903c2c9a53b3f8a66cce7f1da877e0"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_files(base: Path, entries: tuple[Path, ...]) -> list[Path]:
    files: list[Path] = []
    for entry in entries:
        source = base / entry
        if not source.exists():
            raise FileNotFoundError(f"required transfer entry is missing: {source}")
        if source.is_symlink():
            raise ValueError(f"symbolic links are not accepted: {source}")
        if source.is_file():
            files.append(entry)
            continue
        for candidate in source.rglob("*"):
            if candidate.is_symlink():
                raise ValueError(f"symbolic links are not accepted: {candidate}")
            if candidate.is_file():
                files.append(candidate.relative_to(base))
    return sorted(set(files), key=lambda item: item.as_posix())


def validate_critical_files(base: Path) -> None:
    failures: list[str] = []
    for relative, expected in CRITICAL_SHA256.items():
        path = base / Path(relative)
        if not path.is_file():
            failures.append(f"missing: {relative}")
        else:
            actual = sha256_file(path)
            if actual != expected:
                failures.append(f"hash mismatch: {relative} ({actual})")
    if failures:
        raise ValueError("critical artifact validation failed:\n" + "\n".join(failures))


def git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def selected_entries(
    include_features: bool, include_dataset: bool = False
) -> tuple[Path, ...]:
    entries = list(REQUIRED_ENTRIES)
    if include_dataset:
        entries.append(OPTIONAL_DATASET)
    if include_features:
        entries.append(OPTIONAL_FEATURE_CACHE)
    return tuple(entries)


def print_plan(include_features: bool, include_dataset: bool = False) -> None:
    entries = selected_entries(include_features, include_dataset)
    files = iter_files(ROOT, entries)
    total = sum((ROOT / path).stat().st_size for path in files)
    print(f"files: {len(files):,}")
    print(f"bytes: {total:,} ({total / 1024**3:.2f} GiB)")
    for entry in entries:
        print(entry.as_posix())


def create_bundle(
    destination: Path, include_features: bool, include_dataset: bool = False
) -> None:
    destination = destination.resolve()
    if ROOT == destination or ROOT in destination.parents:
        raise ValueError("bundle destination must be outside the Git workspace")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"bundle destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    validate_critical_files(ROOT)
    files = iter_files(ROOT, selected_entries(include_features, include_dataset))
    records: list[dict[str, object]] = []
    for index, relative in enumerate(files, start=1):
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(
            {
                "path": relative.as_posix(),
                "size": source.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
        if index % 5000 == 0:
            print(f"copied and hashed {index:,}/{len(files):,} files", flush=True)

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_head": git_head(),
        "include_dataset": include_dataset,
        "include_feature_cache": include_features,
        "file_count": len(records),
        "total_bytes": sum(int(record["size"]) for record in records),
        "files": records,
    }
    manifest_path = destination / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"bundle ready: {destination}")
    print(f"manifest: {manifest_path}")


def load_manifest(bundle: Path) -> dict[str, object]:
    manifest_path = bundle / MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("files"), list):
        raise ValueError(f"unsupported transfer manifest: {manifest_path}")
    return payload


def safe_relative(raw: object) -> Path:
    if not isinstance(raw, str):
        raise ValueError("manifest path must be a string")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe manifest path: {raw}")
    return relative


def verify_bundle(bundle: Path) -> dict[str, object]:
    bundle = bundle.resolve()
    manifest = load_manifest(bundle)
    failures: list[str] = []
    records = manifest["files"]
    assert isinstance(records, list)
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            failures.append(f"invalid record at index {index}")
            continue
        relative = safe_relative(record.get("path"))
        path = bundle / relative
        if not path.is_file():
            failures.append(f"missing: {relative.as_posix()}")
        elif path.stat().st_size != record.get("size"):
            failures.append(f"size mismatch: {relative.as_posix()}")
        elif sha256_file(path) != record.get("sha256"):
            failures.append(f"hash mismatch: {relative.as_posix()}")
        if index % 5000 == 0:
            print(f"verified {index:,}/{len(records):,} files", flush=True)
    if failures:
        preview = "\n".join(failures[:20])
        raise ValueError(f"bundle verification failed ({len(failures)} errors):\n{preview}")
    print(f"bundle verified: {len(records):,} files")
    return manifest


def restore_bundle(bundle: Path) -> None:
    bundle = bundle.resolve()
    manifest = verify_bundle(bundle)
    records = manifest["files"]
    assert isinstance(records, list)
    copied = 0
    reused = 0
    for record in records:
        assert isinstance(record, dict)
        relative = safe_relative(record["path"])
        source = bundle / relative
        target = ROOT / relative
        if target.exists():
            if (
                target.is_file()
                and target.stat().st_size == record["size"]
                and sha256_file(target) == record["sha256"]
            ):
                reused += 1
                continue
            raise FileExistsError(f"refusing to overwrite different file: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    validate_critical_files(ROOT)
    print(f"restore complete: copied {copied:,}, reused {reused:,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="show files and estimated bundle size")
    plan.add_argument("--include-features", action="store_true")
    plan.add_argument("--include-dataset", action="store_true")

    create = subparsers.add_parser("create", help="create a new private USB bundle")
    create.add_argument("destination", type=Path)
    create.add_argument("--include-features", action="store_true")
    create.add_argument("--include-dataset", action="store_true")

    verify = subparsers.add_parser("verify", help="verify every file in a bundle")
    verify.add_argument("bundle", type=Path)

    restore = subparsers.add_parser("restore", help="restore a bundle into this clone")
    restore.add_argument("bundle", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "plan":
        print_plan(args.include_features, args.include_dataset)
    elif args.command == "create":
        create_bundle(args.destination, args.include_features, args.include_dataset)
    elif args.command == "verify":
        verify_bundle(args.bundle)
    elif args.command == "restore":
        restore_bundle(args.bundle)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
