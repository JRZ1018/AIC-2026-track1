"""Batch 3 queue: continue the duration axis and draw more seeds from the best recipes.

By this point the measured facts are: the online seed noise is +-0.164, min_weight 0.10 ->
0.05 at 12 epochs was worth +0.3685 (above the noise), and training loss kept falling through
20 epochs (0.2324 at 12, 0.2009 at 16, 0.1740 at 20), so duration is the one lever whose
historical effect dwarfs the noise.

  AD  20 epochs, seed 20260912  - a second draw from the longest-duration recipe
  AE  min_weight 0.05 at 24 epochs - best floor with duration pushed further still

The guard probes for ANY running trainer, not just this queue's members.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
LOGS = ROOT / "artifacts" / "queues" / "20260919-batch3"

D = "configs/experiment/"
E = "artifacts/experiments/"

QUEUE = [
    ("AD", D + "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix75_20ep_full_seed2_ad.yaml",
     E + "20260919-ad-final-h1init-t2-last4-lr175e6-cwh-mw10-magmix75-20ep-full-seed2",
     "20 epochs at a fresh seed: second draw from the longest-duration recipe"),
    ("AE", D + "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw05_magmix75_24ep_full_seed1_ae.yaml",
     E + "20260919-ae-final-h1init-t2-last4-lr175e6-cwh-mw05-magmix75-24ep-full-seed1",
     "min_weight 0.05 with 24 epochs: best floor, longest duration"),
]


def _log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with (LOGS / "chain.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _other_training_directories() -> list[str]:
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "Where-Object { $_.CommandLine -like '*train_peft.py*' } | "
                "ForEach-Object { $_.CommandLine }",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as error:  # pragma: no cover - defensive
        _log(f"could not probe running trainers ({error}); assuming none")
        return []
    found = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if "train_peft.py" not in line:
            continue
        if any(output in line for _tag, _config, output, _purpose in QUEUE):
            continue
        found.append(line[:160])
    return found


def _in_flight() -> list[str]:
    pending = []
    for _tag, _config, output, _purpose in QUEUE:
        directory = ROOT / output
        if directory.is_dir() and not (directory / "metrics.json").is_file():
            pending.append(output)
    pending.extend(_other_training_directories())
    return pending


def _wait_for_predecessors() -> None:
    pending = _in_flight()
    if not pending:
        return
    _log("waiting for an in-flight training run: " + ", ".join(pending))
    while True:
        pending = _in_flight()
        if not pending:
            _log("in-flight run finished; proceeding")
            return
        time.sleep(60)


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    _log("batch 3 queue start")
    for tag, config, output, purpose in QUEUE:
        if (ROOT / output / "metrics.json").is_file():
            _log(f"{tag}: already complete, skipping ({purpose})")
            continue
        _wait_for_predecessors()
        log_path = LOGS / f"{tag}.log"
        _log(f"{tag}: starting - {purpose}")
        started = time.time()
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [str(PYTHON), "scripts/train_peft.py", "--config", config],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        elapsed = time.time() - started
        if (ROOT / output / "metrics.json").is_file():
            _log(f"{tag}: finished in {elapsed / 60:.1f} min, metrics.json written")
        else:
            _log(
                f"{tag}: STOPPED after {elapsed / 60:.1f} min (return code {completed.returncode}) "
                f"with NO metrics.json; halting. Log: {log_path}"
            )
            return 1
    _log("batch 3 queue finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
