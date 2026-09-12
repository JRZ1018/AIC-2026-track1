"""Run the locked full-train H1 head and fixed-epoch LoRA final retrain."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return document.get("experiment", document)


def _run_stage(script: str, config: Path, output: Path, artifact: str) -> None:
    selected = output / artifact
    if selected.is_file():
        copied = output / "config.yaml"
        if not copied.is_file() or _sha256(copied) != _sha256(config):
            raise ValueError(f"completed {script} output does not match its locked config")
        return
    command = [sys.executable, str(ROOT / "scripts" / script), "--config", str(config)]
    resume = output / "last.pt"
    if output.exists():
        if not resume.is_file():
            raise FileExistsError(f"partial output lacks resumable last.pt: {output}")
        command.extend(["--resume", str(resume)])
    subprocess.run(command, cwd=ROOT, check=True)
    if not selected.is_file():
        raise RuntimeError(f"{script} did not create {selected}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-config", required=True, type=Path)
    parser.add_argument("--peft-config", required=True, type=Path)
    args = parser.parse_args()
    head_config = args.head_config.resolve()
    peft_config = args.peft_config.resolve()
    head = _load(head_config)
    peft = _load(peft_config)

    for name, raw, epochs in (("head", head, 100), ("peft", peft, 2)):
        if raw.get("training_scope") != "full_train_fixed_epochs":
            raise ValueError(f"{name} config must use full_train_fixed_epochs")
        if int(raw.get("expected_train_count", -1)) != 103218:
            raise ValueError(f"{name} config must require all 103218 training records")
        if int(raw["training"]["epochs"]) != epochs:
            raise ValueError(f"{name} config has an unexpected fixed epoch count")
        if int(raw["training"]["seed"]) != 20260910:
            raise ValueError(f"{name} config must use preregistered seed 20260910")

    head_output = _resolve(head_config.parent, str(head["output_dir"]))
    peft_output = _resolve(peft_config.parent, str(peft["output_dir"]))
    expected_head = (head_output / "final.pt").resolve()
    configured_head = _resolve(peft_config.parent, str(peft["head_checkpoint"]))
    if configured_head != expected_head:
        raise ValueError("PEFT head checkpoint is not the output of the locked H1 stage")
    if peft.get("peft") != {
        "mode": "lora_last_attention",
        "lora_rank": 4,
        "lora_alpha": 8.0,
    }:
        raise ValueError("final PEFT recipe differs from the accepted rank-4 LoRA recipe")
    if float(peft["training"]["head_learning_rate"]) != 0.0:
        raise ValueError("final PEFT classifier head must remain frozen")

    _run_stage("train_frozen_head.py", head_config, head_output, "final.pt")
    _run_stage("train_peft.py", peft_config, peft_output, "final.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
