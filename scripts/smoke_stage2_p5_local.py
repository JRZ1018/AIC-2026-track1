"""Disposable 64-image Stage2 train AMP smoke. Never save learned weights."""
import csv
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aictrack1.data.contracts import load_dataset_config
from aictrack1.data.class_mapping import ClassMapping
from aictrack1.data.indexed_images import IndexedImage, IndexedImageDataset
from aictrack1.data.transforms import build_split_transforms
from aictrack1.engine.training_queue import TrainingLease
from aictrack1.models.clip_vit_b32 import load_clip_vit_b32
from aictrack1.models.clip_resolution import enable_interpolated_visual_resolution
from aictrack1.models.heads import build_classifier_head
from aictrack1.models.peft import configure_visual_peft, PEFTImageClassifier
from aictrack1.methods.sce import symmetric_cross_entropy
from aictrack1.methods.gce import generalized_cross_entropy
from run_stage2_p5_local_fold1_pair import CONFIGS, LOGS, input_hashes, write_json


def main():
    frozen = input_hashes()
    ex = yaml.safe_load(CONFIGS[0].read_text())["experiment"]
    config = load_dataset_config((CONFIGS[0].parent / ex["dataset_config"]).resolve())
    lease = TrainingLease.acquire(ROOT / "artifacts/queues/training.lock.json", owner="p5-local-disposable-smoke",
                                  config=str(CONFIGS[0]), output_dir=str(LOGS / "cuda_smoke.json"), command=tuple(sys.argv))
    try:
        torch.manual_seed(20260921)
        clip_model, _, metadata = load_clip_vit_b32(ROOT / "artifacts/weights/ViT-B-32.pt",
                      frozen["artifacts/weights/ViT-B-32.pt"], "cuda")
        visual = clip_model.visual.float()
        del clip_model
        enable_interpolated_visual_resolution(visual, target_resolution=336)
        configure_visual_peft(visual, "last_n_blocks", lora_blocks=8)
        head_path = (CONFIGS[0].parent / ex["head_checkpoint"]).resolve()
        checkpoint = torch.load(head_path, map_location="cpu", weights_only=True)
        head_meta = checkpoint["feature_cache_metadata"]
        if (checkpoint["competition_stage"] != "stage2" or head_meta["training_scope"] != "fixed_split_validation"
                or head_meta["training_record_count"] != 118958 or head_meta["split_sha256"] != frozen["artifacts/audits/stage2/train/split.csv"]
                or head_meta["clip_weight_sha256"] != metadata.weight_sha256
                or head_meta["class_mapping_sha256"] != ClassMapping.from_csv(config.class_mapping).digest()):
            raise RuntimeError("H1 provenance mismatch")
        head = build_classifier_head(checkpoint["head"]["kind"], 512, 750)
        head.load_state_dict(checkpoint["model"], strict=True)
        for parameter in head.parameters(): parameter.requires_grad_(False)
        model = PEFTImageClassifier(visual, head, normalize_features=head_meta["normalized"]).cuda().train()
        with (ROOT / "artifacts/audits/stage2/train/split.csv").open(encoding="utf-8-sig", newline="") as handle:
            assignments = {r["sample_id"]: r["split"] for r in csv.DictReader(handle)}
        records = []
        with (ROOT / "artifacts/audits/stage2/train/samples.csv").open(encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                if row["source_split"] != "train" or row["decode_status"] != "ok": raise RuntimeError("invalid train record")
                if assignments[row["sample_id"]] != "train": continue
                absolute = (config.sources["train"].root / row["relative_path"]).resolve()
                if not absolute.is_relative_to(config.sources["train"].root): raise RuntimeError("path escape")
                records.append(IndexedImage(index, row["sample_id"], row["relative_path"], absolute, "train", "train", int(row["internal_class_index"]), row["official_class_id"]))
                if len(records) == 64: break
        if len(records) != 64: raise RuntimeError("64 train samples required")
        dataset = IndexedImageDataset(records, build_split_transforms(config.transforms)["train"], stochastic_seed=20260921)
        batch = next(iter(DataLoader(dataset, batch_size=64, num_workers=0)))
        images, labels = batch["image"].cuda(), batch["label"].cuda()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1.75e-6, weight_decay=.01)
        scaler = torch.amp.GradScaler("cuda")
        torch.cuda.reset_peak_memory_stats()
        results = []
        for name, loss_fn in (("gce", lambda z, y: generalized_cross_entropy(z, y, q=.7)),
                              ("sce", lambda z, y: symmetric_cross_entropy(z, y, alpha=.1, beta=1, log_zero=-4))):
            skipped = 0
            for attempt in range(8):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16): loss = loss_fn(model(images), labels)
                if not torch.isfinite(loss): raise FloatingPointError("nonfinite smoke loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if not torch.isfinite(norm):
                    if scaler.get_scale() >= previous_scale: raise FloatingPointError("nonfinite gradient without AMP backoff")
                    skipped += 1
                    continue
                break
            else: raise FloatingPointError("no finite gradient after bounded AMP backoff")
            results.append({"kind": name, "loss": float(loss.item()), "pre_clip_gradient_norm": float(norm.item()),
                            "startup_overflow_skipped_steps": skipped, "final_scaler_scale": scaler.get_scale()})
        result = {"scope": "64 Stage2 train images only; disposable model; two AMP steps; not effect comparison",
                  "passed": True, "samples": 64, "device": torch.cuda.get_device_name(0),
                  "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2, "results": results,
                  "learned_checkpoint_saved": False}
        write_json(LOGS / "cuda_smoke.json", result)
        print(json.dumps(result), flush=True)
    finally:
        lease.release()
    return 0


if __name__ == "__main__": raise SystemExit(main())
