"""Run the 2026-09-18 night queue: determinism check, then the duration axis.

Context: the seed ruler measured the online noise floor at +-0.164, which is the same order
as every recent single-run gain. So this queue stops chasing sub-noise deltas and instead
(a) settles a cheap measurement-technique question whose answer decides whether repeating a
recipe is worth anything at all, and (b) pushes the one axis whose historical effect was far
larger than the noise.

Order:
  V  repeat the 61.0005 config with the SAME seed   - is the online score deterministic?
  W  20 epochs at the best settings                 - the largest lever, one step beyond T
  X  min_weight 0.02                                - middle of the 0.10-0.05 interval, if it matters
  Y  min_weight 0.05 with 16 epochs                 - the two strongest levers together
  Z  mix 0.50 seed lottery (new seed)               - hedged draw against the 60.7922 recipe

Each run is skipped when its metrics.json already exists, and the chain waits for any run
that looks mid-flight, so an interruption resumes rather than repeating work.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
LOGS = ROOT / "artifacts" / "queues" / "20260918-night"

D = "configs/experiment/"
E = "artifacts/experiments/"

QUEUE = [
    ("V", D + "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix75_12ep_full_seed1_v.yaml",
     E + "20260918-v-final-h1init-t2-last4-lr175e6-cwh-mw10-magmix75-12ep-full-seed1-repeat",
     "repeat of the 61.0005 config, same seed: is the online score deterministic?"),
    ("W", D + "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix75_20ep_full_seed1_w.yaml",
     E + "20260918-w-final-h1init-t2-last4-lr175e6-cwh-mw10-magmix75-20ep-full-seed1",
     "20 epochs: the largest historical lever, one step beyond the 16-epoch run"),
    ("X", D + "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw02_magmix75_12ep_full_seed1_x.yaml",
     E + "20260918-x-final-h1init-t2-last4-lr175e6-cwh-mw02-magmix75-12ep-full-seed1",
     "min_weight 0.02: middle of the 0.10-0.05 interval"),
    ("Y", D + "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw05_magmix75_16ep_full_seed1_y.yaml",
     E + "20260918-y-final-h1init-t2-last4-lr175e6-cwh-mw05-magmix75-16ep-full-seed1",
     "min_weight 0.05 with 16 epochs: both strong levers together"),
    ("Z", D + "p3_stage1_h1init_t2_last4_lr175e6_cwh_mw10_magmix50_12ep_full_seed3_z.yaml",
     E + "20260918-z-final-h1init-t2-last4-lr175e6-cwh-mw10-magmix50-12ep-full-seed3",
     "seed lottery on the mix 0.50 recipe, hedged against a wrong mix choice"),
]


def _log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with (LOGS / "chain.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _in_flight() -> list[str]:
    pending = []
    for _tag, _config, output, _purpose in QUEUE:
        directory = ROOT / output
        if directory.is_dir() and not (directory / "metrics.json").is_file():
            pending.append(output)
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
    _log("night queue start")
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
    _log("night queue finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
