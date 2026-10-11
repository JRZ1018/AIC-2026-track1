"""Audit a frozen class-balanced inference prior on train-derived logits only.

The input bundles are produced by ``diagnose_stage2_tta.py`` and are required
to contain Stage 2 train-derived validation logits.  No test manifest or test
image path is accepted by this tool.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import TrainingLease  # noqa: E402
from aictrack1.evaluation.balanced_prior import fit_uniform_soft_bias  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _balanced_proxy_indices(
    targets: torch.Tensor,
    *,
    num_classes: int,
    samples_per_class: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pieces = []
    for class_index in range(num_classes):
        members = torch.nonzero(targets.eq(class_index), as_tuple=False).flatten()
        if members.numel() == 0:
            raise ValueError(f"validation is missing class {class_index}")
        draws = torch.randint(
            members.numel(), (samples_per_class,), generator=generator
        )
        pieces.append(members[draws])
    return torch.cat(pieces)


def _top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float(logits.argmax(dim=1).eq(targets).float().mean().item())


def _load_bundle(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "aictrack1.stage2_train_derived_fused_logits.v1":
        raise ValueError(f"unsupported logits bundle schema: {path}")
    if "train-derived" not in str(payload.get("scope", "")):
        raise ValueError(f"logits bundle is not train-derived validation: {path}")
    logits = payload["logits"]
    targets = payload["targets"]
    train_counts = payload["train_class_counts"]
    if logits.ndim != 2 or targets.ndim != 1 or train_counts.ndim != 1:
        raise ValueError(f"invalid logits bundle tensor ranks: {path}")
    if logits.shape[0] != targets.shape[0] or logits.shape[1] != train_counts.shape[0]:
        raise ValueError(f"inconsistent logits bundle tensor shapes: {path}")
    if not torch.isfinite(logits.float()).all():
        raise ValueError(f"non-finite logits in bundle: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-bundle", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--proxy-seeds", type=int, nargs="+", default=[20260928, 20260929, 20260930])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.samples_per_class < 1 or not args.proxy_seeds:
        raise ValueError("invalid balanced proxy settings")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(args.device)
    lease = None
    if device.type == "cuda":
        lease = TrainingLease.acquire(
            ROOT / "artifacts" / "queues" / "training.lock.json",
            owner="stage2-balanced-prior-train-derived-audit",
            config=";".join(str(path) for path in args.logits_bundle),
            output_dir=str(output),
            command=tuple(sys.argv),
        )
        atexit.register(lease.release)
    temperatures = (0.5, 1.0, 2.0)
    strengths = (0.25, 0.5, 0.75, 1.0)
    prior_strengths = (0.1, 0.25, 0.5, 0.75, 1.0)
    bundles_report = []
    aggregate: dict[str, list[float]] = {}
    for bundle_path in args.logits_bundle:
        bundle_path = bundle_path.resolve()
        payload = _load_bundle(bundle_path)
        logits_cpu = payload["logits"].float()
        targets_cpu = payload["targets"].long()
        train_counts = payload["train_class_counts"].float().clamp_min(1)
        num_classes = logits_cpu.shape[1]
        candidates: dict[str, list[float]] = {"baseline": []}
        diagnostics: dict[str, list[dict[str, float | int]]] = {}
        for seed in args.proxy_seeds:
            indices = _balanced_proxy_indices(
                targets_cpu,
                num_classes=num_classes,
                samples_per_class=args.samples_per_class,
                seed=seed,
            )
            proxy_logits = logits_cpu[indices].to(device)
            proxy_targets = targets_cpu[indices].to(device)
            candidates["baseline"].append(_top1(proxy_logits, proxy_targets))
            log_prior = train_counts.log().to(device)
            log_prior = log_prior - log_prior.mean()
            for strength in prior_strengths:
                name = f"train_prior_subtract_{strength:g}"
                candidates.setdefault(name, []).append(
                    _top1(proxy_logits - strength * log_prior, proxy_targets)
                )
            for temperature in temperatures:
                bias, info = fit_uniform_soft_bias(
                    proxy_logits,
                    temperature_multiplier=temperature,
                    iterations=60,
                )
                diagnostics.setdefault(f"uniform_soft_t{temperature:g}", []).append(info)
                for strength in strengths:
                    name = f"uniform_soft_t{temperature:g}_s{strength:g}"
                    candidates.setdefault(name, []).append(
                        _top1(proxy_logits + strength * bias, proxy_targets)
                    )
        summarized = {}
        baseline_values = candidates["baseline"]
        for name, values in candidates.items():
            deltas = [value - base for value, base in zip(values, baseline_values, strict=True)]
            summarized[name] = {
                "accuracies": values,
                "mean_accuracy": sum(values) / len(values),
                "deltas_vs_baseline": deltas,
                "mean_delta": sum(deltas) / len(deltas),
                "min_delta": min(deltas),
            }
            aggregate.setdefault(name, []).append(summarized[name]["mean_delta"])
        bundles_report.append(
            {
                "path": str(bundle_path),
                "sha256": _sha256(bundle_path),
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "split_sha256": payload["split_sha256"],
                "manifest_sha256": payload["manifest_sha256"],
                "class_mapping_sha256": payload["class_mapping_sha256"],
                "shape": list(logits_cpu.shape),
                "proxy_candidates": summarized,
                "uniform_bias_diagnostics": diagnostics,
            }
        )
    aggregate_summary = {
        name: {
            "bundle_mean_deltas": values,
            "mean_delta": sum(values) / len(values),
            "min_bundle_delta": min(values),
        }
        for name, values in aggregate.items()
    }
    eligible = {
        name: values
        for name, values in aggregate_summary.items()
        if name != "baseline"
        and values["mean_delta"] >= 0.003
        and values["min_bundle_delta"] >= 0.001
    }
    selected = max(eligible, key=lambda name: eligible[name]["mean_delta"]) if eligible else None
    report = {
        "schema": "aictrack1.stage2_balanced_prior_audit.v1",
        "scope": "Stage 2 train-derived balanced proxy only; no test access",
        "samples_per_class": args.samples_per_class,
        "proxy_seeds": args.proxy_seeds,
        "candidate_grid": {
            "train_prior_strengths": prior_strengths,
            "uniform_soft_temperature_multipliers": temperatures,
            "uniform_soft_bias_strengths": strengths,
        },
        "gate": {
            "mean_delta_at_least": 0.003,
            "every_bundle_mean_delta_at_least": 0.001,
        },
        "bundles": bundles_report,
        "aggregate": aggregate_summary,
        "selected_candidate": selected,
        "gate_passed": selected is not None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    if lease is not None:
        lease.release()
        atexit.unregister(lease.release)
    print(json.dumps({"selected_candidate": selected, "gate_passed": selected is not None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
