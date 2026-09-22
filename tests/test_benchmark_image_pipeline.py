from __future__ import annotations

import torch
from torch.utils.data import Dataset

from scripts.benchmark_image_pipeline import _measure_dataset


class _TinyImageDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self.epoch = -1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return 16

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"image": torch.full((3, 4, 4), float(index))}


def test_measure_dataset_reports_non_wrapping_cpu_throughput() -> None:
    dataset = _TinyImageDataset()
    results = _measure_dataset(
        dataset,
        batch_size=2,
        workers=0,
        prefetch_factor=2,
        persistent_workers=True,
        seed=17,
        warmup_batches=1,
        measure_batches=3,
        repeats=2,
        device=torch.device("cpu"),
    )
    assert dataset.epoch == 0
    assert len(results) == 2
    assert all(row["measured_samples"] == 6 for row in results)
    assert all(row["samples_per_second"] > 0 for row in results)
