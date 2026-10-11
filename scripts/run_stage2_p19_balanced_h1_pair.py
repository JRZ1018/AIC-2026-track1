"""Serial, locked P19 head-only screening; configs remain disabled until approved."""

from __future__ import annotations

import sys
import subprocess
import time
from pathlib import Path
from typing import IO, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import QueueItem, run_serial_queue  # noqa: E402


HARD_LIMIT_SECONDS = 20 * 60


def bounded_launcher(deadline: float) -> Callable[[Sequence[str], Path, IO[str], Callable[[int], None]], int]:
    """Stop only this queue's own trainer at the preapproved wall-clock limit."""

    def launch(command: Sequence[str], cwd: Path, output: IO[str], on_start: Callable[[int], None]) -> int:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 124
        process = subprocess.Popen(list(command), cwd=cwd, stdout=output, stderr=subprocess.STDOUT)
        on_start(process.pid)
        try:
            return process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=15)
            output.write("\nP19 hard 20-minute deadline reached; owned trainer terminated at epoch boundary checkpoint.\n")
            output.flush()
            return 124

    return launch


def main() -> int:
    deadline = time.monotonic() + HARD_LIMIT_SECONDS
    return run_serial_queue(
        root=ROOT,
        python=Path(sys.executable),
        logs=ROOT / "artifacts" / "queues" / "20261001-p19-balanced-h1-fold2-fold3",
        queue_name="20261001-p19-balanced-h1-fold2-fold3",
        items=[
            QueueItem(
                "fold2-balanced-softmax",
                "configs/experiment/stage2_p19_fold2_h1_balanced_softmax_336_seed20260921.yaml",
                "artifacts/experiments/20261001-stage2-p19-fold2-h1-balanced-softmax-336-seed20260921",
                "Stage2 train-derived fold2 balanced-softmax linear head",
                trainer_entrypoint="scripts/train_frozen_head.py",
            ),
            QueueItem(
                "fold3-balanced-softmax",
                "configs/experiment/stage2_p19_fold3_h1_balanced_softmax_336_seed20260921.yaml",
                "artifacts/experiments/20261001-stage2-p19-fold3-h1-balanced-softmax-336-seed20260921",
                "Stage2 train-derived fold3 balanced-softmax linear head",
                trainer_entrypoint="scripts/train_frozen_head.py",
            ),
        ],
        launcher=bounded_launcher(deadline),
    )


if __name__ == "__main__":
    raise SystemExit(main())
