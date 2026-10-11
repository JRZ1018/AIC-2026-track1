"""Serial Stage 2 train-derived TTA diagnostics for the preregistered C6 gate."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "artifacts" / "queues" / "20260925-stage2-c6-tta-validation"
DEADLINE = datetime(2026, 9, 25, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
CASES = (
    (
        "fold1-c3-local",
        "configs/experiment/stage2_fold1_c3_last8_ac_8ep_local_confirm_seed20260921.yaml",
        "artifacts/experiments/20260925-stage2-fold1-c3-last8-ac-8ep-local-confirm-seed20260921/best.pt",
        "e2102a3f8bfe27f69ad3a8f5e835e06b8c26d85125c3818420782e8311acd289",
    ),
    (
        "fold1-c5-336",
        "configs/experiment/stage2_fold1_c5_336_last8_ac_8ep_seed20260921.yaml",
        "artifacts/experiments/20260925-stage2-fold1-c5-336-last8-ac-8ep-seed20260921/best.pt",
        "dde01c0fff269c0e977a5e79a4a469a1358e1b3e28ca00f85b1e61d1bbb8437e",
    ),
    (
        "fold2-c5-336",
        "configs/experiment/stage2_fold2_c5_336_last8_ac_8ep_seed20260921.yaml",
        "artifacts/experiments/20260925-stage2-fold2-c5-336-last8-ac-8ep-seed20260921/best.pt",
        "c4542b1459eedc76ccc64c70cde645f72fd94e6837e9e496420d580f7d28f0b3",
    ),
)


def log(message: str) -> None:
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    with (QUEUE / "chain.log").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"[{stamp}] {message}\n")


def main() -> int:
    if QUEUE.exists():
        raise FileExistsError(f"C6 queue already exists: {QUEUE}")
    for tag, config, checkpoint, _ in CASES:
        if not (ROOT / config).is_file() or not (ROOT / checkpoint).is_file():
            raise FileNotFoundError(f"missing C6 case input: {tag}")
        output = ROOT / "artifacts" / "diagnostics" / f"20260925-stage2-c6-{tag}-tta-full"
        if output.exists():
            raise FileExistsError(f"fresh C6 case output already exists: {output}")
    QUEUE.mkdir(parents=True, exist_ok=False)
    log("C6 train-derived validation queue started; frozen C3 fold2 report is reused")
    for tag, config, checkpoint, digest in CASES:
        if datetime.now(ZoneInfo("Asia/Shanghai")) >= DEADLINE:
            raise TimeoutError("C6 deadline reached before next case")
        output = ROOT / "artifacts" / "diagnostics" / f"20260925-stage2-c6-{tag}-tta-full"
        command = [
            sys.executable, str(ROOT / "scripts" / "diagnose_stage2_tta.py"),
            "--experiment-config", str(ROOT / config),
            "--checkpoint", str(ROOT / checkpoint),
            "--expected-checkpoint-sha256", digest,
            "--output-dir", str(output),
            "--batch-size", "64", "--workers", "4",
        ]
        log(f"{tag}: start")
        with (QUEUE / f"{tag}.log").open("w", encoding="utf-8", newline="\n") as handle:
            result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            log(f"{tag}: failed with exit code {result.returncode}")
            return result.returncode
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        if report["smoke_only"] or report["evaluated_count"] != 29737:
            raise RuntimeError(f"{tag}: incomplete validation report")
        log(f"{tag}: complete")
    log("C6 validation queue complete; apply docs/116 gate before any full training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
