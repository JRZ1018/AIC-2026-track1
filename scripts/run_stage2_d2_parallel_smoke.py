"""Run the approved one-epoch L20 controlled-concurrency gate."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.engine.training_queue import QueueItem, run_parallel_queue  # noqa: E402


ITEMS = [
    QueueItem(
        "parallel-a",
        "configs/experiment/runtime_benchmark_stage2_parallel_a_w16_p4_b64_1ep.yaml",
        "artifacts/experiments/runtime-benchmark-stage2-parallel-a-w16-p4-b64-1ep",
        "controlled concurrency gate A",
    ),
    QueueItem(
        "parallel-b",
        "configs/experiment/runtime_benchmark_stage2_parallel_b_w16_p4_b64_1ep.yaml",
        "artifacts/experiments/runtime-benchmark-stage2-parallel-b-w16-p4-b64-1ep",
        "controlled concurrency gate B",
    ),
]


def main() -> int:
    return run_parallel_queue(
        root=ROOT,
        python=Path(sys.executable),
        logs=ROOT / "artifacts/queues/20260923-stage2-d2-parallel-smoke",
        queue_name="20260923-stage2-d2-parallel-smoke",
        items=ITEMS,
        max_concurrent_jobs=2,
    )


if __name__ == "__main__":
    raise SystemExit(main())
