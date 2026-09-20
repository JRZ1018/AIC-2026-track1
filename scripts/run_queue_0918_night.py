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

All historical items use the shared atomic training lock. Validated completed outputs are
skipped; partial outputs, external trainers, probe failures, and stale locks stop the queue.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import QueueItem, run_serial_queue  # noqa: E402

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


def main() -> int:
    return run_serial_queue(
        root=ROOT,
        python=PYTHON,
        logs=LOGS,
        queue_name="20260918-night",
        items=[QueueItem(*item) for item in QUEUE],
    )


if __name__ == "__main__":
    raise SystemExit(main())
