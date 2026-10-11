"""Train-only empirical-marginal visual OT feasibility audit; no model training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import signal
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aictrack1.data.feature_cache import load_feature_cache  # noqa: E402
from aictrack1.engine.training_queue import TrainingLease, read_training_lock, probe_running_trainers  # noqa: E402
from aictrack1.methods.confidence import classwise_rank_noise_scores  # noqa: E402

CACHE = ROOT / "artifacts/features/stage2/p15_train_official336_20260930"
OUTPUT = ROOT / "artifacts/reports/20260930-p15-stage2-visual-ot-feasibility-attempt2.json"
QUEUE = ROOT / "artifacts/queues/20260930-p15-visual-ot-audit-attempt2"
LOCK = Path("/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json")
MANIFEST = ROOT / "artifacts/audits/stage2/train/samples.csv"
FOLD2 = ROOT / "artifacts/audits/stage2/train/split_fold2_seeded_random_20260924.csv"
FOLD3 = ROOT / "artifacts/audits/stage2/train/split_fold3_seeded_random_20260925.csv"
CONFLICTS = ROOT / "artifacts/audits/stage2/train/label_conflicts.csv"
EXPECTED = {
    MANIFEST: "1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d",
    FOLD2: "1e5eb7e9adbc234ba933bf76c99896602b8bdce5557879f90224ae2bb4cf2a50",
    FOLD3: "dcef8ccb919fec62d451fe78089d11399e7de2aaf779df7c636cdbdf176cdf68",
    ROOT / "artifacts/weights/ViT-B-32.pt": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
}
TAU = 0.05
ITERATIONS = 30
FRACTION = 0.10


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_labels(path: Path, sample_ids: tuple[str, ...]) -> torch.Tensor:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = {row["sample_id"]: row["split"] for row in csv.DictReader(stream)}
    if len(rows) != len(sample_ids) or set(rows) != set(sample_ids):
        raise ValueError(f"split identifiers do not exactly match train cache: {path}")
    values = [rows[key] for key in sample_ids]
    if set(values) != {"train", "validation"}:
        raise ValueError(f"unexpected split labels in {path}")
    return torch.tensor([value == "train" for value in values], dtype=torch.bool)


def conflict_flags(sample_ids: tuple[str, ...], labels: torch.Tensor) -> torch.Tensor:
    with CONFLICTS.open("r", encoding="utf-8-sig", newline="") as stream:
        conflict_hashes = {row["sha256"] for row in csv.DictReader(stream)}
    rows: dict[str, tuple[str, int]] = {}
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            rows[row["sample_id"]] = (row["sha256"], int(row["internal_class_index"]))
    if len(rows) != len(sample_ids) or set(rows) != set(sample_ids):
        raise ValueError("manifest identifiers do not exactly match train cache")
    if any(rows[key][1] != int(labels[i]) for i, key in enumerate(sample_ids)):
        raise ValueError("manifest labels do not match cached train labels")
    return torch.tensor([rows[key][0] in conflict_hashes for key in sample_ids], dtype=torch.bool)


def m2_low_trust(target: torch.Tensor, margin: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    noise = 0.5 * classwise_rank_noise_scores(-target, labels) + 0.5 * classwise_rank_noise_scores(-margin, labels)
    selected = torch.zeros(len(labels), dtype=torch.bool)
    for category in range(750):
        positions = torch.nonzero(labels == category).flatten()
        count = max(1, round(len(positions) * FRACTION))
        chosen = positions[torch.argsort(noise[positions], descending=True, stable=True)[:count]]
        selected[chosen] = True
    return selected


def analyze_fold(name: str, features: torch.Tensor, labels: torch.Tensor, conflicts: torch.Tensor, mask: torch.Tensor) -> dict:
    positions = torch.nonzero(mask).flatten()
    if len(positions) != 118958 or len(mask) - len(positions) != 29737:
        raise ValueError(f"{name} expected 118958/29737 train-derived split")
    y = labels[positions].long()
    x = F.normalize(features[positions].float().to("cuda"), dim=1)
    n = len(y)
    counts = torch.bincount(y, minlength=750)
    if bool((counts < 2).any()):
        raise ValueError(f"{name} a class has fewer than two train references")
    y_gpu = y.to("cuda")
    counts_gpu = counts.float().to("cuda")
    sums = torch.zeros((750, x.shape[1]), dtype=torch.float32, device="cuda")
    sums.index_add_(0, y_gpu, x)
    prototypes = F.normalize(sums / counts_gpu[:, None], dim=1)
    similarities = x @ prototypes.T
    target_loo = (x * F.normalize(sums[y_gpu] - x, dim=1)).sum(1)
    similarities[torch.arange(n, device="cuda"), y_gpu] = target_loo
    other = similarities.clone()
    other[torch.arange(n, device="cuda"), y_gpu] = -torch.inf
    margin = target_loo - other.max(1).values
    kernel = torch.exp((similarities / TAU) - (similarities / TAU).max(1, keepdim=True).values)
    desired = counts_gpu / n
    scaling = torch.ones(750, dtype=torch.float32, device="cuda")
    for _ in range(ITERATIONS):
        denominator = (kernel @ scaling).clamp_min(1e-20)
        empirical = ((kernel * scaling) / denominator[:, None]).mean(0).clamp_min(1e-20)
        scaling = (scaling * desired / empirical).clamp(1e-12, 1e12)
    denominator = (kernel @ scaling).clamp_min(1e-20)
    posterior = (kernel[torch.arange(n, device="cuda"), y_gpu] * scaling[y_gpu] / denominator).cpu()
    empirical = ((kernel * scaling) / denominator[:, None]).mean(0)
    marginal_l1 = float((empirical - desired).abs().sum().item())
    if not torch.isfinite(posterior).all() or not math.isfinite(marginal_l1) or marginal_l1 > 0.05:
        raise RuntimeError(f"{name} OT marginal/numerical convergence failed: L1={marginal_l1}")
    target_cpu = target_loo.cpu()
    margin_cpu = margin.cpu()
    baseline = m2_low_trust(target_cpu, margin_cpu, y)
    count = round(n * FRACTION)
    ot = torch.zeros(n, dtype=torch.bool)
    ot[torch.argsort(posterior, stable=True)[:count]] = True
    conflict = conflicts[positions]
    prevalence = float(conflict.float().mean())
    def enrichment(selection: torch.Tensor) -> float:
        return float(conflict[selection].float().mean()) / prevalence
    ordered = torch.argsort(counts, stable=True)
    head = torch.isin(y, ordered[-150:])
    tail = torch.isin(y, ordered[:375])
    head_rate = float(ot[head].float().mean())
    tail_rate = float(ot[tail].float().mean())
    overlap = int((ot & baseline).sum())
    union = int((ot | baseline).sum())
    jaccard = overlap / union
    ot_enrichment = enrichment(ot)
    m2_enrichment = enrichment(baseline)
    tail_head_ratio = tail_rate / head_rate if head_rate > 0 else None
    passed = jaccard < 0.8 and ot_enrichment > m2_enrichment and tail_head_ratio is not None and tail_head_ratio <= 2.0
    return {
        "fold": name, "train_count": n, "validation_count": len(mask) - n,
        "class_support_min": int(counts.min()), "class_support_median": float(counts.float().median()), "class_support_max": int(counts.max()),
        "conflict_prevalence": prevalence, "ot_selected": int(ot.sum()), "m2_selected": int(baseline.sum()),
        "ot_conflict_enrichment": ot_enrichment, "m2_conflict_enrichment": m2_enrichment,
        "ot_m2_lowtrust_jaccard": jaccard, "ot_tail_lowtrust_rate": tail_rate,
        "ot_head_lowtrust_rate": head_rate, "ot_tail_head_rate_ratio": tail_head_ratio,
        "ot_target_posterior_mean": float(posterior.mean()), "sinkhorn_marginal_l1": marginal_l1,
        "feasibility_pass": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if sys.platform != "linux" or ROOT != Path("/root/autodl-tmp/AICtrack1_p14_anchor01"):
        raise RuntimeError("P15 audit requires exact test-free server workspace")
    if (ROOT / "DataSetStage2/test").exists():
        raise RuntimeError("P15 audit refuses any workspace containing test")
    if read_training_lock(LOCK) is not None or probe_running_trainers():
        raise RuntimeError("shared lock/trainer active")
    if not torch.cuda.is_available() or "RTX 4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("server RTX4090 unavailable")
    if OUTPUT.exists() or QUEUE.exists():
        raise FileExistsError("P15 audit already exists; no automatic retry")
    if not (CACHE / "complete.json").is_file():
        raise FileNotFoundError("P15 complete feature cache missing")
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise RuntimeError(f"P15 source SHA mismatch: {path}")
    if shutil.disk_usage(ROOT).free < 6 * 1024**3:
        raise RuntimeError("P15 audit disk floor failed")
    if not args.run:
        print(json.dumps({"scope": "Stage2 train only", "cache": str(CACHE), "fold2_sha": EXPECTED[FOLD2], "fold3_sha": EXPECTED[FOLD3], "gpu": torch.cuda.get_device_name(0)}))
        return 0
    lease = TrainingLease.acquire(LOCK, owner=QUEUE.name, config="docs/163_P15长尾视觉OT可行性审计预注册_20260930.md", output_dir=str(OUTPUT), command=sys.argv)
    try:
        def timeout_handler(_signum, _frame):
            raise TimeoutError("P15 OT audit exceeded 20-minute hard cutoff")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(20 * 60)
        QUEUE.mkdir(parents=True, exist_ok=False)
        (QUEUE / "source_receipt.json").write_text(json.dumps({"started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "queue_pid": os.getpid(), "lock_token": lease.record.token, "input_sha256": {str(path.relative_to(ROOT)): digest for path, digest in EXPECTED.items()}, "cache_complete_sha256": sha(CACHE / "complete.json")}, indent=2) + "\n", encoding="utf-8")
        cache = load_feature_cache(CACHE)
        if cache.metadata.source != "train" or cache.metadata.dataset_stage != "rematch" or cache.metadata.dataset_config_sha256 != "a3736c041aa8fb1b2a5126a0842597f94d4b834bc84a41af90cc2768bf19cafe" or cache.metadata.expected_record_count != 148695 or cache.metadata.feature_dim != 512 or not cache.metadata.normalized or cache.metadata.manifest_sha256 != EXPECTED[MANIFEST] or cache.metadata.split_sha256 != EXPECTED[FOLD2] or cache.metadata.clip_weight_sha256 != EXPECTED[ROOT / "artifacts/weights/ViT-B-32.pt"]:
            raise RuntimeError("P15 feature cache metadata mismatch")
        if len(set(cache.sample_ids)) != 148695 or len(cache.labels) != 148695 or int(cache.labels.min()) < 0 or int(cache.labels.max()) >= 750:
            raise RuntimeError("P15 cached row/label contract mismatch")
        norms = torch.linalg.vector_norm(cache.features.float(), dim=1)
        if not torch.isfinite(norms).all() or not torch.allclose(norms, torch.ones_like(norms), atol=2e-3, rtol=2e-3):
            raise RuntimeError("P15 normalized feature values invalid")
        flags = conflict_flags(cache.sample_ids, cache.labels)
        fold2 = torch.tensor([value == "train" for value in cache.partitions], dtype=torch.bool)
        if not torch.equal(fold2, split_labels(FOLD2, cache.sample_ids)):
            raise RuntimeError("P15 fold2 cache partition mismatch")
        fold3 = split_labels(FOLD3, cache.sample_ids)
        rows = [analyze_fold("fold2", cache.features, cache.labels, flags, fold2), analyze_fold("fold3", cache.features, cache.labels, flags, fold3)]
        report = {"schema": "aictrack1.p15_visual_ot_feasibility.v1", "scope": "Stage2 official train only; no test, no training", "ot": {"temperature": TAU, "iterations": ITERATIONS, "column_marginal": "empirical_fold_train_class_frequency", "selected_global_fraction": FRACTION}, "cache_complete_sha256": sha(CACHE / "complete.json"), "folds": rows, "feasibility_pass": all(row["feasibility_pass"] for row in rows), "decision": "eligible_for_preregistered_peft_screen" if all(row["feasibility_pass"] for row in rows) else "stop_no_distinct_signal"}
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (QUEUE / "queue_result.json").write_text(json.dumps({"exit_code": 0, "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "report_sha256": sha(OUTPUT)}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"decision": report["decision"], "folds": rows, "report_sha256": sha(OUTPUT)}, indent=2))
        return 0
    finally:
        signal.alarm(0)
        lease.release()


if __name__ == "__main__":
    raise SystemExit(main())
