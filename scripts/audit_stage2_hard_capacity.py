"""Frozen P10 hard-capacity screen using only Stage 2 train-derived logits."""
from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.class_mapping import ClassMapping  # noqa: E402
from aictrack1.engine.training_queue import TrainingLease  # noqa: E402
from aictrack1.evaluation.balanced_prior import fit_uniform_soft_bias  # noqa: E402
from aictrack1.evaluation.feature_geometry import grouped_bootstrap_delta  # noqa: E402
from aictrack1.evaluation.hard_capacity import decode_hard_capacity  # noqa: E402
from aictrack1.evaluation.long_tail import stratify_classes  # noqa: E402

TRAIN = ROOT / "artifacts" / "audits" / "stage2" / "train"
AUDIT = ROOT / "artifacts" / "audits" / "stage2"
OUTPUT = ROOT / "artifacts" / "diagnostics" / "20260929-p10-hard-capacity-proxies"
REPORT_PATH = ROOT / "artifacts" / "reports" / "20260929-p10-hard-capacity-screen.json"
MANIFEST_SHA = "1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d"
MAPPING_DIGEST = "811b767e5689fbed5ed9ed7f328e2e1458b02b208647f3ec1b3c3e99c7e4e879"
MAPPING_FILE_SHA = "0f54c471945f356177c885ba2f9edc943d669a61f12c73503fd4632c91ef40be"
COUNTS_SHA = "103db2edc63e10bf3d0b8173c339fe0646404412ec8bf277cbccc84103ab7a09"
CONFLICTS_SHA = "b1afe084e654aeac5d431501657734c168a3cfc1f928defacac219dcf84c430a"
SEEDS = (20260928, 20260929, 20260930)
PER_CLASS = 50
TEMP_MULTIPLIER = 0.5
BIAS_STRENGTH = 0.75
BOOTSTRAP_SEED = 20260925
BOOTSTRAP_REPS = 500
WALL_LIMIT = 3600
CPU_LIMIT = 2700
RAM_LIMIT = 4 * 1024**3
DISK_LIMIT = 200 * 1024**2
MIN_DISK_FREE = 3 * 1024**3
MIN_GPU_FREE = 1024**3

FOLDS = (
    {
        "fold": 2,
        "directory": "20260928-p7-fold2-p5-sce-fourview-logits",
        "split_file": "split_fold2_seeded_random_20260924.csv",
        "logits_sha": "0aecb9b15eec86b781fa8d98c98d78ccb14eb636404fa7cfd86146cd9166c06d",
        "paired_sha": "13353c60e2511b9228affcddd3f6d5c8ab43ee5f64c945983dc0b4225046aba3",
        "report_sha": "bd38a3969efd342c14c53efa3f0b6add4898747f4aa698ba0a0e3ebd2d249c06",
        "checkpoint_sha": "abe02be63f09bc85381b2867e639a6d7b0295b5752d9c26ddc7422f0ec512ffe",
        "config_sha": "c888c74c993f42148ee955a36400842b01b0ddca7b5e76ce07372931c8472b9c",
        "split_sha": "1e5eb7e9adbc234ba933bf76c99896602b8bdce5557879f90224ae2bb4cf2a50",
    },
    {
        "fold": 3,
        "directory": "20260928-p7-fold3-p5-sce-fourview-logits",
        "split_file": "split_fold3_seeded_random_20260925.csv",
        "logits_sha": "c0e9209f52bc7b7b44e2a322935c3452f2250bd34af4c8b0a2cc274f49319a1e",
        "paired_sha": "3b7f74620d29707a3855299dc15f9bcaa6412f0caff4628d78fbc0890b15cee4",
        "report_sha": "86f6134ac1a58b87661dfc28120fabc57848f3b57ecc01cec793ae64b7f6e7b0",
        "checkpoint_sha": "2c550da87a7301d386f0b78cf4319635c8db9f3a65a07b4951047c53bb26f060",
        "config_sha": "5b48db475a8c3b4ae315b2c5405eb1edba7fbe61f80111deeabb5948716aff5a",
        "split_sha": "dcef8ccb919fec62d451fe78089d11399e7de2aaf779df7c636cdbdf176cdf68",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def balanced_indices(targets: torch.Tensor, *, classes: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pieces = []
    for index in range(classes):
        members = torch.nonzero(targets.eq(index), as_tuple=False).flatten()
        if not members.numel():
            raise ValueError(f"validation split is missing class {index}")
        pieces.append(torch.randint(members.numel(), (PER_CLASS,), generator=generator))
        pieces[-1] = members[pieces[-1]]
    return torch.cat(pieces)


def load_train_manifest() -> tuple[dict[str, dict[str, str]], set[str]]:
    manifest_path = TRAIN / "samples.csv"
    conflicts_path = TRAIN / "label_conflicts.csv"
    if sha256(manifest_path) != MANIFEST_SHA or sha256(conflicts_path) != CONFLICTS_SHA:
        raise ValueError("Stage 2 train manifest/conflicts SHA-256 mismatch")
    samples: dict[str, dict[str, str]] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "source_split", "internal_class_index", "sha256", "near_duplicate_group"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("train manifest is missing required columns")
        for row in reader:
            if row["sample_id"] in samples or row["source_split"] != "train":
                raise ValueError("train manifest contains duplicate IDs or non-train rows")
            samples[row["sample_id"]] = row
    if len(samples) != 148_695:
        raise ValueError(f"expected 148695 train rows, found {len(samples)}")
    with conflicts_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"sha256", "label_count"}.issubset(reader.fieldnames):
            raise ValueError("train conflict report is missing columns")
        conflict_hashes = {row["sha256"] for row in reader if int(row["label_count"]) > 1}
    return samples, conflict_hashes


def read_split(
    path: Path, expected_sha: str, samples: dict[str, dict[str, str]]
) -> tuple[set[str], list[int]]:
    if sha256(path) != expected_sha:
        raise ValueError(f"fixed train split SHA mismatch: {path}")
    validation_ids: set[str] = set()
    train_counts = [0] * 750
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["sample_id", "split"]:
            raise ValueError("train split CSV schema mismatch")
        for row in reader:
            sample_id, split = row["sample_id"], row["split"]
            if sample_id in seen or sample_id not in samples:
                raise ValueError("train split has duplicate or unknown IDs")
            seen.add(sample_id)
            if split == "validation":
                validation_ids.add(sample_id)
            elif split == "train":
                train_counts[int(samples[sample_id]["internal_class_index"])] += 1
            else:
                raise ValueError(f"unknown split value: {split!r}")
    if len(seen) != 148_695 or len(validation_ids) != 29_737:
        raise ValueError("fixed train split row counts mismatch")
    return validation_ids, train_counts


def load_fold(
    spec: dict[str, Any],
    samples: dict[str, dict[str, str]],
    conflict_hashes: set[str],
    mapping_digest: str,
) -> dict[str, Any]:
    directory = ROOT / "artifacts" / "diagnostics" / spec["directory"]
    bundle_path, paired_path, report_path = (
        directory / "fused_logits.pt",
        directory / "paired_predictions.csv",
        directory / "report.json",
    )
    for path, expected in (
        (bundle_path, spec["logits_sha"]),
        (paired_path, spec["paired_sha"]),
        (report_path, spec["report_sha"]),
    ):
        if sha256(path) != expected:
            raise ValueError(f"frozen P7 train-derived input SHA mismatch: {path}")
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_report = {
        "scope": "Stage 2 train-derived fixed validation only; no test access",
        "smoke_only": False,
        "full_validation_count": 29_737,
        "checkpoint_sha256": spec["checkpoint_sha"],
        "experiment_config_sha256": spec["config_sha"],
        "split_sha256": spec["split_sha"],
        "manifest_sha256": MANIFEST_SHA,
        "class_mapping_sha256": MAPPING_DIGEST,
    }
    if any(source_report.get(key) != value for key, value in expected_report.items()):
        raise ValueError("P7 source report provenance mismatch")
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    if (
        bundle.get("schema") != "aictrack1.stage2_train_derived_fused_logits.v1"
        or bundle.get("scope") != expected_report["scope"]
        or bundle.get("checkpoint_sha256") != spec["checkpoint_sha"]
        or bundle.get("experiment_config_sha256") != spec["config_sha"]
        or bundle.get("split_sha256") != spec["split_sha"]
        or bundle.get("manifest_sha256") != MANIFEST_SHA
        or bundle.get("class_mapping_sha256") != mapping_digest
        or tuple(bundle["logits"].shape) != (29_737, 750)
        or tuple(bundle["targets"].shape) != (29_737,)
        or tuple(bundle["train_class_counts"].shape) != (750,)
    ):
        raise ValueError("P7 fused-logits bundle provenance or shape mismatch")
    logits, targets = bundle["logits"], bundle["targets"].long()
    train_counts = bundle["train_class_counts"].long()
    if not torch.isfinite(logits.float()).all() or (train_counts <= 0).any():
        raise ValueError("fused logits or train class counts are invalid")
    split_path = TRAIN / spec["split_file"]
    validation_ids, split_train_counts = read_split(split_path, spec["split_sha"], samples)
    if train_counts.tolist() != split_train_counts:
        raise ValueError("bundle train class counts disagree with the fixed split")

    with paired_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_columns = [
            "sample_id", "target", "center_prediction", "flip_tta_prediction", "four_view_prediction"
        ]
        if reader.fieldnames != expected_columns:
            raise ValueError("paired-prediction CSV schema mismatch")
        paired = list(reader)
    ids = [row["sample_id"] for row in paired]
    if len(ids) != 29_737 or len(set(ids)) != 29_737 or set(ids) != validation_ids:
        raise ValueError("paired predictions do not match fixed train validation IDs")
    groups: list[str] = []
    conflicts: list[bool] = []
    mismatches = 0
    rounded_argmax = logits.argmax(dim=1).tolist()
    for index, row in enumerate(paired):
        sample = samples[row["sample_id"]]
        target = int(row["target"])
        if target != int(targets[index]) or target != int(sample["internal_class_index"]):
            raise ValueError(f"paired target mismatch at row {index}")
        mismatches += int(int(row["four_view_prediction"]) != rounded_argmax[index])
        group = sample["near_duplicate_group"].strip()
        groups.append(f"near:{group}" if group else f"sample:{row['sample_id']}")
        conflicts.append(sample["sha256"] in conflict_hashes)
    return {
        "fold": spec["fold"],
        "spec": spec,
        "bundle": bundle,
        "logits": logits,
        "targets": targets,
        "sample_ids": ids,
        "groups": groups,
        "conflicts": conflicts,
        "fp16_argmax_mismatches": mismatches,
    }


def proxy_metrics(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    targets: torch.Tensor,
    *,
    strata: tuple[str, ...],
    conflicts: list[bool],
    groups: list[str],
) -> dict[str, Any]:
    base_ok, candidate_ok = baseline.eq(targets), candidate.eq(targets)
    classes = len(strata)
    supports = torch.bincount(targets.cpu(), minlength=classes).float()
    if not torch.equal(supports, torch.full_like(supports, PER_CLASS)):
        raise ValueError("proxy does not contain exactly 50 samples per class")
    base_correct = torch.bincount(targets.cpu()[base_ok.cpu()], minlength=classes).float()
    candidate_correct = torch.bincount(
        targets.cpu()[candidate_ok.cpu()], minlength=classes
    ).float()
    base_recall, candidate_recall = base_correct / supports, candidate_correct / supports
    tail = [i for i, value in enumerate(strata) if value == "tail"]
    if not tail:
        raise ValueError("class stratification contains no tail classes")
    conflict = torch.tensor(conflicts, dtype=torch.bool)
    ordinary = ~conflict
    base_cpu, candidate_cpu = base_ok.cpu(), candidate_ok.cpu()
    base_values = base_cpu.numpy().astype("float64")
    candidate_values = candidate_cpu.numpy().astype("float64")
    bootstrap = grouped_bootstrap_delta(
        base_values,
        candidate_values,
        groups,
        seed=BOOTSTRAP_SEED,
        replicates=BOOTSTRAP_REPS,
    )

    def subset_accuracy(values: torch.Tensor, mask: torch.Tensor) -> float | None:
        return float(values[mask].float().mean().item()) if bool(mask.any()) else None

    base_metrics = {
        "top1": float(base_ok.float().mean().item()),
        "macro_accuracy": float(base_recall.mean().item()),
        "tail_class_mean_recall": float(base_recall[tail].mean().item()),
        "ordinary_top1": subset_accuracy(base_cpu, ordinary),
        "conflict_top1": subset_accuracy(base_cpu, conflict),
    }
    candidate_metrics = {
        "top1": float(candidate_ok.float().mean().item()),
        "macro_accuracy": float(candidate_recall.mean().item()),
        "tail_class_mean_recall": float(candidate_recall[tail].mean().item()),
        "ordinary_top1": subset_accuracy(candidate_cpu, ordinary),
        "conflict_top1": subset_accuracy(candidate_cpu, conflict),
    }
    deltas = {
        key: (
            None
            if base_metrics[key] is None or candidate_metrics[key] is None
            else (candidate_metrics[key] - base_metrics[key]) * 100.0
        )
        for key in base_metrics
    }
    return {
        "sample_count": int(targets.numel()),
        "per_class_capacity": PER_CLASS,
        "baseline": base_metrics,
        "candidate": candidate_metrics,
        "deltas_pp": deltas,
        "label_conflict_rows": int(conflict.sum()),
        "paired_wins": int((candidate_ok & ~base_ok).sum().item()),
        "paired_losses": int((base_ok & ~candidate_ok).sum().item()),
        "paired_group_bootstrap_delta_pp": {
            key: (float(value) * 100.0 if key in {"delta", "ci95_low", "ci95_high"} else value)
            for key, value in bootstrap.items()
        },
    }


def write_proxy_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    fields = (
        "proxy_row",
        "sample_id",
        "target",
        "near_duplicate_group",
        "label_conflict",
        "baseline_prediction",
        "capacity_prediction",
    )
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty metric")
    return float(sum(values) / len(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if OUTPUT.exists() or REPORT_PATH.exists() or REPORT_PATH.with_suffix(
        REPORT_PATH.suffix + ".tmp"
    ).exists():
        raise FileExistsError("P10 outputs already exist; refusing to overwrite")
    disk_before = shutil.disk_usage(ROOT)
    if disk_before.free < MIN_DISK_FREE:
        raise RuntimeError("free disk is below the preregistered 3 GiB minimum")
    device = torch.device(args.device)
    gpu: dict[str, Any] = {"device": str(device)}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        if free < MIN_GPU_FREE:
            raise RuntimeError("free GPU memory is below the preregistered 1 GiB minimum")
        torch.cuda.reset_peak_memory_stats(device)
        gpu.update(
            {
                "name": torch.cuda.get_device_name(device),
                "free_bytes_before": int(free),
                "total_bytes": int(total),
            }
        )

    lease = TrainingLease.acquire(
        ROOT / "artifacts" / "queues" / "training.lock.json",
        owner="stage2-p10-hard-capacity-train-derived-screen",
        config=(
            "fixed P7 soft bias t=0.5 strength=0.75; proxy seeds 20260928-30; "
            "50 samples/class; capacity 50/class; no test access"
        ),
        output_dir=str(OUTPUT),
        command=tuple(sys.argv),
    )
    atexit.register(lease.release)
    started, cpu_started = time.monotonic(), time.process_time()
    report: dict[str, Any] = {
        "schema": "aictrack1.stage2_p10_hard_capacity_screen.v1",
        "status": "running",
        "scope": "Stage 2 train-derived balanced proxies only; no test access",
        "algorithm": {
            "baseline": "frozen P7 uniform soft bias fit on each proxy",
            "candidate": "same adjusted logits with deterministic greedy class capacities",
            "temperature_multiplier": TEMP_MULTIPLIER,
            "bias_strength": BIAS_STRENGTH,
            "bias_iterations": 60,
            "proxy_samples_per_class": PER_CLASS,
            "proxy_capacity_lower": PER_CLASS,
            "proxy_capacity_upper": PER_CLASS,
            "proxy_seeds": list(SEEDS),
            "tie_break": ["regret ascending", "proxy row ascending", "class index ascending"],
        },
        "input_hashes": {},
        "source_hashes": {
            "audit_script_sha256": sha256(Path(__file__)),
            "decoder_sha256": sha256(ROOT / "src/aictrack1/evaluation/hard_capacity.py"),
            "balanced_prior_sha256": sha256(ROOT / "src/aictrack1/evaluation/balanced_prior.py"),
            "long_tail_sha256": sha256(ROOT / "src/aictrack1/evaluation/long_tail.py"),
            "feature_geometry_sha256": sha256(
                ROOT / "src/aictrack1/evaluation/feature_geometry.py"
            ),
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": gpu,
        },
        "resource_budget": {
            "wall_limit_seconds": WALL_LIMIT,
            "cpu_limit_seconds": CPU_LIMIT,
            "ram_limit_bytes": RAM_LIMIT,
            "disk_limit_bytes": DISK_LIMIT,
            "gpu_cumulative_limit_seconds": 900,
            "min_gpu_free_bytes": MIN_GPU_FREE,
            "min_disk_free_bytes": MIN_DISK_FREE,
            "estimated_peak_extra_ram_upper_bound_bytes": 2 * 1024**3,
        },
        "folds": [],
    }
    gpu_cumulative_seconds = 0.0
    try:
        OUTPUT.mkdir(parents=True, exist_ok=False)
        mapping_path = AUDIT / "metadata" / "classes.csv"
        counts_path = TRAIN / "class_counts.csv"
        manifest_path = TRAIN / "samples.csv"
        conflicts_path = TRAIN / "label_conflicts.csv"
        mapping = ClassMapping.from_csv(mapping_path)
        if (
            mapping.digest() != MAPPING_DIGEST
            or sha256(mapping_path) != MAPPING_FILE_SHA
            or sha256(counts_path) != COUNTS_SHA
            or len(mapping.entries) != 750
        ):
            raise ValueError("Stage 2 train class mapping/count source mismatch")
        samples, conflict_hashes = load_train_manifest()
        report["input_hashes"].update(
            {
                "train_manifest_sha256": sha256(manifest_path),
                "train_class_counts_sha256": sha256(counts_path),
                "train_label_conflicts_sha256": sha256(conflicts_path),
                "class_mapping_file_sha256": sha256(mapping_path),
                "class_mapping_digest": mapping.digest(),
            }
        )

        for spec in FOLDS:
            if time.monotonic() - started > WALL_LIMIT:
                raise TimeoutError("P10 wall-clock budget exceeded")
            fold = load_fold(spec, samples, conflict_hashes, mapping.digest())
            report["input_hashes"][f"fold{spec['fold']}"] = {
                "fused_logits_sha256": spec["logits_sha"],
                "paired_predictions_sha256": spec["paired_sha"],
                "source_report_sha256": spec["report_sha"],
                "checkpoint_sha256": spec["checkpoint_sha"],
                "config_sha256": spec["config_sha"],
                "split_sha256": spec["split_sha"],
            }
            seed = 20260924 if spec["fold"] == 2 else 20260925
            stratification = stratify_classes(
                fold["bundle"]["train_class_counts"],
                head_fraction=0.2,
                mid_fraction=0.3,
                seed=seed,
            )
            fold_result: dict[str, Any] = {
                "fold": spec["fold"],
                "validation_rows": 29_737,
                "train_class_counts_sum": int(fold["bundle"]["train_class_counts"].sum()),
                "class_strata": {
                    "head": stratification.head_count,
                    "mid": stratification.mid_count,
                    "tail": int(stratification.counts["tail"]),
                },
                "fp16_argmax_mismatches_vs_saved_full_precision_predictions": (
                    fold["fp16_argmax_mismatches"]
                ),
                "proxies": [],
            }
            fold["logits"] = fold["logits"].float()
            for proxy_seed in SEEDS:
                if time.monotonic() - started > WALL_LIMIT:
                    raise TimeoutError("P10 wall-clock budget exceeded")
                gpu_interval_started = time.monotonic() if device.type == "cuda" else None
                indices = balanced_indices(
                    fold["targets"], classes=750, seed=proxy_seed
                )
                proxy_logits = fold["logits"][indices].to(device)
                proxy_targets = fold["targets"][indices].to(device)
                bias, bias_info = fit_uniform_soft_bias(
                    proxy_logits,
                    temperature_multiplier=TEMP_MULTIPLIER,
                    iterations=60,
                )
                adjusted = proxy_logits + BIAS_STRENGTH * bias
                baseline = adjusted.argmax(dim=1)
                candidate, capacity = decode_hard_capacity(
                    adjusted, lower=PER_CLASS, upper=PER_CLASS
                )
                if not capacity["quota_exact"]:
                    raise RuntimeError("hard capacity proxy failed exact quota")
                selected = indices.tolist()
                proxy_conflicts = [fold["conflicts"][i] for i in selected]
                proxy_groups = [fold["groups"][i] for i in selected]
                metrics = proxy_metrics(
                    baseline,
                    candidate,
                    proxy_targets,
                    strata=stratification.assignments,
                    conflicts=proxy_conflicts,
                    groups=proxy_groups,
                )
                if gpu_interval_started is not None:
                    gpu_cumulative_seconds += time.monotonic() - gpu_interval_started
                    if gpu_cumulative_seconds > 900:
                        raise TimeoutError("P10 cumulative GPU wall budget exceeded")
                prediction_rows = [
                    {
                        "proxy_row": row,
                        "sample_id": fold["sample_ids"][source_index],
                        "target": int(proxy_targets[row].item()),
                        "near_duplicate_group": proxy_groups[row],
                        "label_conflict": int(proxy_conflicts[row]),
                        "baseline_prediction": int(baseline[row].item()),
                        "capacity_prediction": int(candidate[row].item()),
                    }
                    for row, source_index in enumerate(selected)
                ]
                csv_path = OUTPUT / f"fold{spec['fold']}_seed{proxy_seed}.csv"
                write_proxy_csv(csv_path, prediction_rows)
                metrics["capacity"] = capacity
                metrics["uniform_bias_fit"] = bias_info
                metrics["proxy_predictions"] = {
                    "path": str(csv_path.relative_to(ROOT)),
                    "sha256": sha256(csv_path),
                    "size_bytes": csv_path.stat().st_size,
                }
                fold_result["proxies"].append({"seed": proxy_seed, **metrics})
                del proxy_logits, adjusted, baseline, candidate, proxy_targets, indices
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                if time.process_time() - cpu_started > CPU_LIMIT:
                    raise TimeoutError("P10 CPU-time budget exceeded")
                if sum(p.stat().st_size for p in OUTPUT.glob("*.csv")) > DISK_LIMIT:
                    raise RuntimeError("P10 proxy outputs exceed 200 MiB")
            fold_result["mean_deltas_pp"] = {
                key: mean([float(p["deltas_pp"][key]) for p in fold_result["proxies"]])
                for key in ("top1", "macro_accuracy", "tail_class_mean_recall", "ordinary_top1")
            }
            report["folds"].append(fold_result)

        fold_means = [fold["mean_deltas_pp"] for fold in report["folds"]]
        overall = {
            key: mean([float(fold[key]) for fold in fold_means])
            for key in ("top1", "macro_accuracy", "tail_class_mean_recall", "ordinary_top1")
        }
        output_bytes = sum(p.stat().st_size for p in OUTPUT.glob("*.csv"))
        disk_written = max(0, int(disk_before.free - shutil.disk_usage(ROOT).free))
        resource_ok = (
            time.monotonic() - started <= WALL_LIMIT
            and time.process_time() - cpu_started <= CPU_LIMIT
            and output_bytes <= DISK_LIMIT
            and disk_written <= DISK_LIMIT
            and gpu_cumulative_seconds <= 900
            and report["resource_budget"]["estimated_peak_extra_ram_upper_bound_bytes"]
            <= RAM_LIMIT
        )
        checks = {
            "all_six_proxies_complete": sum(len(f["proxies"]) for f in report["folds"]) == 6,
            "all_capacities_exact": all(
                p["capacity"]["quota_exact"] for f in report["folds"] for p in f["proxies"]
            ),
            "mean_top1_delta_pp_at_least_0_30": overall["top1"] >= 0.30,
            "each_fold_top1_delta_pp_at_least_0_10": all(
                f["mean_deltas_pp"]["top1"] >= 0.10 for f in report["folds"]
            ),
            "mean_macro_delta_pp_positive": overall["macro_accuracy"] > 0.0,
            "mean_tail_delta_pp_at_least_minus_0_25": overall["tail_class_mean_recall"] >= -0.25,
            "mean_ordinary_delta_pp_nonnegative": overall["ordinary_top1"] >= 0.0,
            "resource_budget_passed": resource_ok,
        }
        report.update(
            {
                "status": "screen_complete",
                "gpu_cumulative_wall_seconds": gpu_cumulative_seconds,
                "fold_mean_deltas_pp": fold_means,
                "overall_mean_deltas_pp": overall,
                "gate_checks": checks,
                "gate_passed": all(checks.values()),
                "decision": (
                    "eligible_for_one_frozen_test_inference"
                    if all(checks.values())
                    else "stop_p10_keep_existing_p7_submission"
                ),
                "bootstrap": {
                    "method": "paired bootstrap resampling near-duplicate groups",
                    "seed": BOOTSTRAP_SEED,
                    "replicates": BOOTSTRAP_REPS,
                    "ci_crossing_zero_is_not_a_standalone_gate": True,
                },
            }
        )
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["runtime_seconds"] = time.monotonic() - started
        report["cpu_seconds"] = time.process_time() - cpu_started
        report.setdefault("gpu_cumulative_wall_seconds", gpu_cumulative_seconds)
        disk_after = shutil.disk_usage(ROOT)
        report["disk_bytes_written"] = max(0, int(disk_before.free - disk_after.free))
        report["runtime_within_budget"] = (
            report["runtime_seconds"] <= WALL_LIMIT
            and report["cpu_seconds"] <= CPU_LIMIT
            and report["disk_bytes_written"] <= DISK_LIMIT
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            free_after, total_after = torch.cuda.mem_get_info(device)
            gpu.update(
                {
                    "free_bytes_after": int(free_after),
                    "total_bytes_after": int(total_after),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
            )
        report["environment"]["gpu"] = gpu
        report["output_files"] = (
            [
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(OUTPUT.glob("*.csv"))
            ]
            if OUTPUT.exists()
            else []
        )
        try:
            atomic_json(REPORT_PATH, report)
        finally:
            if not lease.released:
                lease.release()
            atexit.unregister(lease.release)

    print(
        json.dumps(
            {
                "status": report["status"],
                "gate_passed": report.get("gate_passed"),
                "decision": report.get("decision"),
                "report": str(REPORT_PATH.relative_to(ROOT)),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
