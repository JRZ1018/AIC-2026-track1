from __future__ import annotations

import gc
import hashlib
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from aictrack1.data.class_mapping import ClassEntry, ClassMapping
from aictrack1.data.indexed_images import IndexedImage, IndexedImageDataset
from aictrack1.engine.provenance import CheckpointProvenance
from aictrack1.evaluation.classification import classification_diagnostics
from aictrack1.methods.confidence import ClasswiseHistoryWeighting
from aictrack1.models.heads import build_classifier_head, load_classifier_head_checkpoint
from aictrack1.submission import (
    SubmissionSpec,
    build_submission_package,
    submission_keys_from_relative_paths,
    validate_submission_zip,
)


ANNOUNCED_STAGE2_TRAIN_COUNT = 148_695
ANNOUNCED_STAGE2_TEST_COUNT = 37_444
ANNOUNCED_STAGE2_CLASS_COUNT = 750
FEATURE_DIM = 16
SHA = {
    "mapping": "1" * 64,
    "manifest": "2" * 64,
    "split": "3" * 64,
    "clip": "4" * 64,
}
RUN_METADATA = {
    "run_id": "stage2-dynamic-scale-regression",
    "checkpoint_sha256": "5" * 64,
    "config_hash": "6" * 64,
    "data_hash": "7" * 64,
    "git_commit": "8" * 40,
    "inference_command": "synthetic-regression-only",
}


def _mapping(num_classes: int) -> ClassMapping:
    return ClassMapping(
        [
            ClassEntry(
                official_class_id=f"cls_{index:05d}",
                internal_class_index=index,
                class_name=f"synthetic class {index}",
            )
            for index in range(num_classes)
        ]
    )


def _sequence_sha256(values: tuple[str, ...] | list[str]) -> str:
    digest = hashlib.sha256()
    for index, value in enumerate(values):
        if index:
            digest.update(b"\n")
        digest.update(value.encode("utf-8"))
    return digest.hexdigest()


@pytest.mark.parametrize("num_classes", [3, 7, 500, 750])
def test_dynamic_class_count_round_trips_head_loss_checkpoint_and_package(
    tmp_path: Path, num_classes: int
) -> None:
    mapping = _mapping(num_classes)
    mapping_path = tmp_path / "classes.csv"
    mapping.write_csv(mapping_path)
    restored_mapping = ClassMapping.from_csv(mapping_path)
    assert len(restored_mapping) == num_classes
    assert restored_mapping.digest() == mapping.digest()

    diagnostic_targets = torch.arange(num_classes)
    diagnostic_predictions = diagnostic_targets.clone()
    diagnostic_predictions[-1] = 0
    diagnostics = classification_diagnostics(
        diagnostic_predictions,
        diagnostic_targets,
        num_classes=num_classes,
    )
    assert diagnostics["present_target_classes"] == num_classes
    assert diagnostics["sample_count"] == num_classes

    batch_size = min(num_classes, 8)
    features = torch.randn(batch_size, FEATURE_DIM)
    targets = torch.arange(batch_size) % num_classes
    for kind in ("linear", "cosine"):
        head = build_classifier_head(kind, FEATURE_DIM, num_classes)
        logits = head(features)
        assert logits.shape == (batch_size, num_classes)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(parameter.grad is not None for parameter in head.parameters())
        assert all(parameter.ndim <= 2 for parameter in head.parameters())
        assert max(parameter.numel() for parameter in head.parameters()) <= (
            num_classes * FEATURE_DIM
        )

    provenance = CheckpointProvenance(
        competition_stage="stage2",
        class_mapping_sha256=SHA["mapping"],
        train_manifest_sha256=SHA["manifest"],
        split_sha256=SHA["split"],
        clip_weight_sha256=SHA["clip"],
        num_classes=num_classes,
    )
    head = build_classifier_head("linear", FEATURE_DIM, num_classes)
    checkpoint_path = tmp_path / "head.pt"
    torch.save(
        {
            **provenance.to_dict(),
            "head": {"kind": "linear"},
            "feature_cache_metadata": {
                "clip_weight_sha256": SHA["clip"],
                "class_mapping_sha256": SHA["mapping"],
                "split_sha256": SHA["split"],
            },
            "model": head.state_dict(),
        },
        checkpoint_path,
    )
    restored_head, payload = load_classifier_head_checkpoint(
        checkpoint_path,
        feature_dim=FEATURE_DIM,
        num_classes=num_classes,
        expected_clip_weight_sha256=SHA["clip"],
        expected_class_mapping_sha256=SHA["mapping"],
        expected_split_sha256=SHA["split"],
        expected_provenance=provenance,
    )
    assert payload["num_classes"] == num_classes
    assert torch.equal(head(features), restored_head(features))

    filenames = [f"test_{index:06d}.jpg" for index in range(num_classes)]
    predictions = {name: index for index, name in enumerate(filenames)}
    package = build_submission_package(
        predictions,
        filenames,
        mapping,
        tmp_path / "submission",
        run_metadata=RUN_METADATA,
    )
    assert package.record_count == num_classes
    validate_submission_zip(
        package.zip_path,
        filenames,
        [entry.official_class_id for entry in mapping.entries],
    )


def _confidence_tensor_bytes(state: ClasswiseHistoryWeighting) -> int:
    tensors = [value for value in vars(state).values() if isinstance(value, torch.Tensor)]
    assert tensors
    assert all(tensor.ndim == 1 for tensor in tensors)
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def test_stage2_confidence_state_is_linear_at_announced_train_scale() -> None:
    small_count = 2_048
    small = ClasswiseHistoryWeighting(
        torch.arange(small_count),
        torch.arange(small_count) % ANNOUNCED_STAGE2_CLASS_COUNT,
    )
    large = ClasswiseHistoryWeighting(
        torch.arange(small_count * 2),
        torch.arange(small_count * 2) % ANNOUNCED_STAGE2_CLASS_COUNT,
    )
    assert _confidence_tensor_bytes(large) == 2 * _confidence_tensor_bytes(small)

    full = ClasswiseHistoryWeighting(
        torch.arange(ANNOUNCED_STAGE2_TRAIN_COUNT),
        torch.arange(ANNOUNCED_STAGE2_TRAIN_COUNT) % ANNOUNCED_STAGE2_CLASS_COUNT,
    )
    full_bytes = _confidence_tensor_bytes(full)
    assert full_bytes < 16 * 1024 * 1024
    assert all(
        tensor.numel() <= ANNOUNCED_STAGE2_TRAIN_COUNT
        for tensor in vars(full).values()
        if isinstance(tensor, torch.Tensor)
    )
    assert all(
        tensor.ndim == 1
        for tensor in full.state_dict().values()
        if isinstance(tensor, torch.Tensor)
    )


def _build_manifest_key_sequences() -> tuple[list[str], tuple[str, ...]]:
    train_paths = [
        f"train/class_{index % ANNOUNCED_STAGE2_CLASS_COUNT:03d}/sample_{index:06d}.jpg"
        for index in range(ANNOUNCED_STAGE2_TRAIN_COUNT - 1, -1, -1)
    ]
    train_paths.sort(key=lambda value: value.encode("utf-8"))
    test_paths = [
        f"test/batch_{index // 1_000:03d}/image_{index:06d}.jpg"
        for index in range(ANNOUNCED_STAGE2_TEST_COUNT)
    ]
    test_keys = submission_keys_from_relative_paths(
        test_paths,
        spec=SubmissionSpec(filename_mode="relative_path"),
    )
    return train_paths, test_keys


def test_announced_stage2_manifest_key_scale_is_deterministic_and_bounded() -> None:
    gc.collect()
    tracemalloc.start()
    train_paths, test_keys = _build_manifest_key_sequences()
    train_digest = _sequence_sha256(train_paths)
    test_digest = _sequence_sha256(test_keys)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert len(train_paths) == ANNOUNCED_STAGE2_TRAIN_COUNT
    assert len(test_keys) == ANNOUNCED_STAGE2_TEST_COUNT
    assert train_paths == sorted(train_paths, key=lambda value: value.encode("utf-8"))
    assert len(set(train_paths)) == len(train_paths)
    assert len({value.casefold() for value in test_keys}) == len(test_keys)
    assert train_digest == (
        "b74e032a581918f48bebd0dd11604568bdca002829c0383cc2933e1f2468ed88"
    )
    assert test_digest == (
        "fdc6b593d1f4aaa35d5499d6635ee255f0526ab0b74433fd2c28685fa388b73f"
    )
    assert peak_bytes < 128 * 1024 * 1024


class _ToyVisual(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(3 * 8 * 8, FEATURE_DIM)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images.flatten(start_dim=1))


def _image_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def test_small_real_image_dataloader_and_750_class_forward(tmp_path: Path) -> None:
    records: list[IndexedImage] = []
    for index in range(4):
        image_path = tmp_path / f"sample_{index}.png"
        Image.new("RGB", (8, 8), color=(index * 40, 20, 200)).save(image_path)
        records.append(
            IndexedImage(
                sample_index=index,
                sample_id=f"sample-{index}",
                relative_path=image_path.name,
                absolute_path=image_path,
                source="train",
                partition="train",
                internal_class_index=(index * 249) % ANNOUNCED_STAGE2_CLASS_COUNT,
                official_class_id=f"cls_{(index * 249) % ANNOUNCED_STAGE2_CLASS_COUNT:05d}",
            )
        )

    dataset = IndexedImageDataset(records, _image_tensor)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    visual = _ToyVisual()
    head = build_classifier_head("linear", FEATURE_DIM, ANNOUNCED_STAGE2_CLASS_COUNT)
    seen = 0
    for batch in loader:
        logits = head(visual(batch["image"]))
        assert logits.shape == (len(batch["label"]), ANNOUNCED_STAGE2_CLASS_COUNT)
        loss = torch.nn.functional.cross_entropy(logits, batch["label"])
        loss.backward()
        assert torch.isfinite(loss)
        seen += len(batch["label"])
    assert seen == len(records)
