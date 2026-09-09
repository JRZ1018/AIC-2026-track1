"""Read-only dataset auditing and deterministic report generation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .class_mapping import ClassMapping
from .contracts import DatasetConfig, ensure_output_path_is_safe
from .duplicates import assign_duplicate_groups
from .manifest import SampleRecord, build_manifest, write_manifest


def _write_dict_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit_records(
    records: list[SampleRecord],
    *,
    near_hamming: int = 3,
    max_bucket_size: int = 500,
) -> tuple[list[SampleRecord], dict[str, Any], list[dict[str, Any]]]:
    grouped, duplicate_diagnostics = assign_duplicate_groups(
        records,
        near_hamming=near_hamming,
        max_bucket_size=max_bucket_size,
    )
    statuses = Counter(record.decode_status for record in grouped)
    class_counts = Counter(record.official_class_id for record in grouped if record.official_class_id)
    sha_labels: dict[str, set[str]] = defaultdict(set)
    for record in grouped:
        if record.sha256 and record.official_class_id:
            sha_labels[record.sha256].add(record.official_class_id)
    conflicts = [
        {"sha256": sha, "labels": ";".join(sorted(labels)), "label_count": len(labels)}
        for sha, labels in sorted(sha_labels.items())
        if len(labels) > 1
    ]
    summary: dict[str, Any] = {
        "sample_count": len(grouped),
        "class_count": len(class_counts),
        "decode_status_counts": dict(sorted(statuses.items())),
        "bad_file_count": sum(count for status, count in statuses.items() if status != "ok"),
        "exact_duplicate_groups": duplicate_diagnostics["exact_groups"],
        "near_duplicate_groups": duplicate_diagnostics["near_groups"],
        "same_content_label_conflicts": len(conflicts),
        **duplicate_diagnostics,
    }
    return grouped, summary, conflicts


def write_audit_report(
    config: DatasetConfig,
    source_name: str,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    records = build_manifest(config, source_name)
    grouped, summary, conflicts = audit_records(
        records,
        near_hamming=config.audit.near_duplicate_hamming,
        max_bucket_size=config.audit.near_duplicate_max_bucket,
    )
    output = ensure_output_path_is_safe(
        config,
        Path(output_root).resolve() if output_root else config.audit.output_root / source_name,
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest_hash = write_manifest(grouped, output / "samples.csv")
    bad_rows = [asdict(record) for record in grouped if record.decode_status != "ok"]
    _write_dict_rows(
        output / "bad_files.csv",
        bad_rows,
        list(asdict(grouped[0]).keys()) if grouped else ["sample_id"],
    )
    class_counts = Counter(record.official_class_id for record in grouped if record.official_class_id)
    _write_dict_rows(
        output / "class_counts.csv",
        [{"official_class_id": label, "count": count} for label, count in sorted(class_counts.items())],
        ["official_class_id", "count"],
    )
    duplicate_rows = [
        {
            "sample_id": record.sample_id,
            "relative_path": record.relative_path,
            "observed_label": record.observed_label,
            "duplicate_group": record.duplicate_group,
            "near_duplicate_group": record.near_duplicate_group,
        }
        for record in grouped
        if record.duplicate_group or record.near_duplicate_group
    ]
    _write_dict_rows(
        output / "duplicate_groups.csv",
        duplicate_rows,
        ["sample_id", "relative_path", "observed_label", "duplicate_group", "near_duplicate_group"],
    )
    _write_dict_rows(output / "label_conflicts.csv", conflicts, ["sha256", "labels", "label_count"])
    summary.update(
        {
            "stage": config.stage,
            "source": source_name,
            "class_mapping_sha256": ClassMapping.from_csv(config.class_mapping).digest(),
            "manifest_sha256": manifest_hash,
            "config_sha256": hashlib.sha256(config.config_path.read_bytes()).hexdigest(),
        }
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "manifest.sha256").write_text(f"{manifest_hash}  samples.csv\n", encoding="utf-8")
    failures = []
    if config.audit.fail_on_bad_files and summary["bad_file_count"]:
        failures.append(f"{summary['bad_file_count']} bad files")
    if config.audit.fail_on_label_conflicts and summary["same_content_label_conflicts"]:
        failures.append(f"{summary['same_content_label_conflicts']} same-content label conflicts")
    if failures:
        raise RuntimeError(f"strict data audit failed ({', '.join(failures)}); report written to {output}")
    return summary
