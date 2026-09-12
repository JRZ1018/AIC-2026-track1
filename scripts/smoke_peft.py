"""Measure one PEFT mode on a bounded train-only stage-1 subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.class_mapping import ClassMapping
from aictrack1.data.contracts import ensure_output_path_is_safe, load_dataset_config
from aictrack1.data.indexed_images import IndexedImageDataset, read_indexed_images
from aictrack1.data.transforms import build_split_transforms
from aictrack1.engine.peft import load_trainable_state_dict, trainable_state_dict
from aictrack1.models.clip_vit_b32 import load_clip_vit_b32
from aictrack1.models.heads import CosineClassifier
from aictrack1.models.peft import PEFTImageClassifier, configure_visual_peft


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_settings(path: Path) -> tuple[Path, str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = raw.get("model", raw)
    if raw.get("provider") != "OpenAI" or raw.get("name") != "ViT-B/32":
        raise ValueError("model config must declare OpenAI ViT-B/32")
    if raw.get("runtime_downloads_allowed") is not False:
        raise ValueError("PEFT smoke requires runtime downloads to be disabled")
    weight = Path(str(raw["weight_path"]))
    resolved = (weight if weight.is_absolute() else path.parent / weight).resolve()
    return resolved, str(raw["expected_weight_sha256"])


@torch.inference_mode()
def _evaluate(model: PEFTImageClassifier, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)
        correct += int(logits.argmax(dim=1).eq(labels).sum())
        count += len(labels)
    return correct / count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "layernorm_projection",
            "last_block",
            "lora_last_attention",
            "lora_last_n_attention",
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-train", type=int, default=256)
    parser.add_argument("--max-validation", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-blocks", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.max_train, args.max_validation, args.batch_size, args.steps) < 1:
        raise ValueError("sample counts, batch size, and steps must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    dataset_config = load_dataset_config(args.dataset_config)
    output = ensure_output_path_is_safe(dataset_config, args.output)
    if output.exists():
        raise FileExistsError(f"PEFT smoke output already exists: {output}")
    output.mkdir(parents=True)
    all_records = read_indexed_images(
        dataset_config,
        source="train",
        manifest_path=args.manifest,
        split_path=args.split,
    )
    train_records = [record for record in all_records if record.partition == "train"][: args.max_train]
    validation_records = [record for record in all_records if record.partition == "validation"][: args.max_validation]
    if not train_records or not validation_records:
        raise ValueError("bounded PEFT smoke selections cannot be empty")
    transforms = build_split_transforms(dataset_config.transforms)
    train_dataset = IndexedImageDataset(
        train_records,
        transforms["train"],
        allow_truncated_images=dataset_config.allow_truncated_images,
        stochastic_seed=args.seed,
    )
    validation_dataset = IndexedImageDataset(
        validation_records,
        transforms["validation"],
        allow_truncated_images=dataset_config.allow_truncated_images,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    weight_path, expected_weight_sha256 = _model_settings(args.model_config.resolve())
    clip_model, _, clip_metadata = load_clip_vit_b32(
        weight_path, expected_weight_sha256, device
    )
    visual = clip_model.visual.float()
    del clip_model
    summary = configure_visual_peft(
        visual,
        args.mode,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_blocks=args.lora_blocks,
    )
    classifier = CosineClassifier(clip_metadata.feature_dim, len(ClassMapping.from_csv(dataset_config.class_mapping)))
    model = PEFTImageClassifier(visual, classifier).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    losses: list[float] = []
    seen = 0
    started = time.perf_counter()
    iterator = iter(train_loader)
    for _ in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            train_dataset.set_epoch(train_dataset.epoch + 1)
            iterator = iter(train_loader)
            batch = next(iterator)
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss = F.cross_entropy(model(images), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("PEFT smoke produced a non-finite loss")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
        seen += len(labels)
    train_seconds = time.perf_counter() - started
    validation_top1 = _evaluate(model, validation_loader, device)

    checkpoint_path = output / "trainable_state.pt"
    state = trainable_state_dict(model)
    torch.save(
        {
            "mode": args.mode,
            "clip_weight_sha256": clip_metadata.weight_sha256,
            "trainable_state": state,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        },
        checkpoint_path,
    )
    probe = next(iter(validation_loader))["image"].to(device)
    model.eval()
    with torch.inference_mode():
        before = model(probe).float().cpu()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.zero_()
    load_trainable_state_dict(model, state)
    with torch.inference_mode():
        after = model(probe).float().cpu()
    if not torch.equal(before, after):
        raise RuntimeError("PEFT trainable checkpoint did not restore exact model outputs")

    result = {
        "scope": "bounded stage1 train/validation PEFT resource smoke; test data not accessed",
        "mode": args.mode,
        "train_samples_available": len(train_records),
        "validation_samples": len(validation_records),
        "steps": args.steps,
        "samples_processed": seen,
        "train_seconds": train_seconds,
        "samples_per_second": seen / train_seconds,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "validation_top1_smoke_only": validation_top1,
        "total_visual_parameters": summary.total_visual_parameters,
        "trainable_visual_parameters": summary.trainable_visual_parameters,
        "trainable_total_parameters": sum(parameter.numel() for parameter in trainable),
        "trainable_parameter_names": list(summary.trainable_parameter_names),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "clip_weight_sha256": clip_metadata.weight_sha256,
        "dataset_config_sha256": _sha256(args.dataset_config.resolve()),
        "manifest_sha256": _sha256(args.manifest.resolve()),
        "split_sha256": _sha256(args.split.resolve()),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "command": " ".join(sys.argv),
        "torch_version": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
