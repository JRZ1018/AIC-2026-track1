"""Extract the single approved Stage2 full-train 336 feature cache on A10."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.manifest import sha256_file  # noqa: E402
from aictrack1.engine.training_queue import TrainingLease  # noqa: E402

CONFIG = ROOT / "configs/experiment/stage2_p5_sce_h1_336_full_a10.yaml"
CACHE = ROOT / "artifacts/features/stage2/train_official336_p5_20260928"
QUEUE = ROOT / "artifacts/queues/20260928-p5-a10-full-features"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        raise RuntimeError("the P5 A10 full feature route is Linux-only")
    if shutil.disk_usage(ROOT).free < 8 * 1024**3:
        raise RuntimeError("the P5 A10 full feature route requires 8 GiB free")
    experiment = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["experiment"]
    if experiment.get("execution_enabled") is not True:
        raise RuntimeError("the full SCE route is not enabled")
    output = (CONFIG.parent / experiment["feature_cache"]).resolve()
    if output != CACHE or experiment.get("expected_train_count") != 148695:
        raise ValueError("unexpected full feature cache identity")
    data = (CONFIG.parent / experiment["dataset_config"]).resolve()
    manifest = ROOT / "artifacts/audits/stage2/train/samples.csv"
    split = ROOT / "artifacts/audits/stage2/train/split.csv"
    model_config = ROOT / "configs/model/openai_clip_vit_b32.yaml"
    model = yaml.safe_load(model_config.read_text(encoding="utf-8"))["model"]
    if (output / "complete.json").is_file():
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        expected = {
            "source": "train",
            "dataset_stage": "stage2",
            "dataset_config_sha256": sha256_file(data),
            "manifest_sha256": sha256_file(manifest),
            "split_sha256": sha256_file(split),
            "clip_weight_sha256": model["expected_weight_sha256"],
            "expected_record_count": 148695,
            "peft_checkpoint_sha256": None,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("existing full feature cache provenance mismatch")
        if metadata.get("preprocessing", {}).get("center_crop") != 336:
            raise ValueError("existing full feature cache resolution mismatch")
        print("Matching complete P5 cache already exists.")
        return 0
    if output.exists() and not args.resume:
        raise FileExistsError("partial cache requires explicit --resume")
    if QUEUE.exists():
        raise FileExistsError("feature queue already exists; inspect before retry")
    lease = TrainingLease.acquire(
        ROOT / "artifacts/queues/training.lock.json",
        owner=QUEUE.name,
        config=str(CONFIG),
        output_dir=str(output),
        command=tuple(sys.argv),
    )
    try:
        QUEUE.mkdir(parents=True, exist_ok=False)
        command = [
            sys.executable,
            "scripts/extract_clip_features.py",
            "--config", str(data),
            "--source", "train",
            "--manifest", str(manifest),
            "--split", str(split),
            "--model-config", str(model_config),
            "--visual-resolution", "336",
            "--device", "cuda",
            "--feature-dtype", "float16",
            "--output", str(output),
            "--batch-size", "64",
            "--workers", "4",
            "--shard-size", "4096",
        ]
        if args.resume:
            command.append("--resume")
        with (QUEUE / "features.log").open("w", encoding="utf-8", newline="\n") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            lease.attach_trainer(child.pid)
            code = child.wait()
        if code != 0:
            raise RuntimeError(f"full feature extraction failed with exit code {code}")
        stats = json.loads((output / "extraction_stats.json").read_text(encoding="utf-8"))
        if stats.get("record_count") != 148695 or not (output / "complete.json").is_file():
            raise RuntimeError("full feature extraction is incomplete")
    finally:
        lease.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
