"""Generate deterministic procedural images and known noisy-label ground truth."""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


NOISE_MODES = {"symmetric", "pairflip", "family", "instance", "class_dependent"}


@dataclass(frozen=True)
class SyntheticConfig:
    output_root: Path
    seed: int = 42
    num_classes: int = 12
    classes_per_family: int = 3
    max_samples_per_class: int = 24
    imbalance_ratio: float = 4.0
    image_size: int = 64
    noise_mode: str = "family"
    noise_rate: float = 0.3
    exact_duplicate_count: int = 4
    near_duplicate_count: int = 4
    corrupt_count: int = 1
    truncated_count: int = 1


def _validate(config: SyntheticConfig) -> None:
    if config.num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if config.classes_per_family < 2:
        raise ValueError("classes_per_family must be at least 2")
    if config.max_samples_per_class < 2:
        raise ValueError("max_samples_per_class must be at least 2")
    if config.imbalance_ratio < 1:
        raise ValueError("imbalance_ratio must be at least 1")
    if config.image_size < 24:
        raise ValueError("image_size must be at least 24")
    if config.noise_mode not in NOISE_MODES:
        raise ValueError(f"unsupported noise mode {config.noise_mode!r}")
    if not 0 <= config.noise_rate <= 1:
        raise ValueError("noise_rate must be in [0, 1]")
    for name in ("exact_duplicate_count", "near_duplicate_count", "corrupt_count", "truncated_count"):
        if getattr(config, name) < 0:
            raise ValueError(f"{name} cannot be negative")


def synthetic_config_from_mapping(raw: dict[str, Any], base: Path) -> SyntheticConfig:
    section = raw.get("synthetic", raw)
    output = Path(str(section["output_root"]))
    output = (output if output.is_absolute() else base / output).resolve()
    kwargs = {key: value for key, value in section.items() if key != "output_root"}
    config = SyntheticConfig(output_root=output, **kwargs)
    _validate(config)
    return config


def class_counts(config: SyntheticConfig) -> list[int]:
    if config.num_classes == 1:
        return [config.max_samples_per_class]
    counts = [
        max(1, round(config.max_samples_per_class * config.imbalance_ratio ** (-index / (config.num_classes - 1))))
        for index in range(config.num_classes)
    ]
    return [min(counts[index], counts[index - 1]) if index else counts[index] for index in range(len(counts))]


def _class_color(class_index: int) -> tuple[int, int, int]:
    palette = (
        (210, 70, 70),
        (70, 160, 210),
        (90, 185, 105),
        (190, 130, 55),
        (155, 90, 195),
        (65, 185, 175),
    )
    return palette[(class_index // 3) % len(palette)]


def _render_image(config: SyntheticConfig, class_index: int, sample_index: int, difficulty: float, rng: np.random.Generator) -> Image.Image:
    size = config.image_size
    background = np.full((size, size, 3), 225, dtype=np.float32)
    noise = rng.normal(0, 7 + 15 * difficulty, background.shape)
    background = np.clip(background + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(background, mode="RGB")
    draw = ImageDraw.Draw(image)
    family = class_index // config.classes_per_family
    variant = class_index % config.classes_per_family
    color = _class_color(class_index)
    jitter = int(rng.integers(-3, 4))
    margin = 10 + jitter
    box = (margin, margin, size - margin, size - margin)
    shape = family % 3
    if shape == 0:
        draw.ellipse(box, fill=color, outline=(30, 30, 30), width=2)
    elif shape == 1:
        draw.rounded_rectangle(box, radius=5 + variant * 2, fill=color, outline=(30, 30, 30), width=2)
    else:
        points = [(size // 2, margin), (size - margin, size - margin), (margin, size - margin)]
        draw.polygon(points, fill=color, outline=(30, 30, 30))
    stripe_step = 7 + variant * 2
    for position in range(margin + stripe_step, size - margin, stripe_step):
        if family % 2:
            draw.line((margin, position, size - margin, position), fill=(245, 245, 245), width=1 + variant % 2)
        else:
            draw.line((position, margin, position, size - margin), fill=(245, 245, 245), width=1 + variant % 2)
    marker_x = margin + 5 + variant * 7
    marker_y = size - margin - 7
    draw.rectangle((marker_x, marker_y, marker_x + 4, marker_y + 4), fill=(20, 20, 20))
    if difficulty > 0:
        image = ImageEnhance.Contrast(image).enhance(1.0 - 0.35 * difficulty)
        image = image.filter(ImageFilter.GaussianBlur(radius=0.8 * difficulty))
    angle = float(rng.normal(0, 2.0 + 3.0 * difficulty))
    return image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(225, 225, 225))


def _alternate_label(config: SyntheticConfig, clean: int, difficulty: float, rng: np.random.Generator) -> int:
    if config.noise_mode == "symmetric":
        value = int(rng.integers(0, config.num_classes - 1))
        return value + int(value >= clean)
    if config.noise_mode == "pairflip":
        return (clean + 1) % config.num_classes
    family_start = (clean // config.classes_per_family) * config.classes_per_family
    family = [value for value in range(family_start, min(family_start + config.classes_per_family, config.num_classes)) if value != clean]
    if family:
        return int(rng.choice(family))
    return (clean + 1) % config.num_classes


def _flip_probability(config: SyntheticConfig, clean: int, difficulty: float) -> float:
    if config.noise_mode == "instance":
        return min(1.0, config.noise_rate * (0.25 + 1.5 * difficulty))
    if config.noise_mode == "class_dependent":
        scale = 0.5 + clean / max(1, config.num_classes - 1)
        return min(1.0, config.noise_rate * scale)
    return config.noise_rate


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def generate_synthetic_dataset(config: SyntheticConfig) -> dict[str, Any]:
    _validate(config)
    if config.output_root.exists() and any(config.output_root.iterdir()):
        raise FileExistsError(f"synthetic output is not empty: {config.output_root}")
    images = config.output_root / "images"
    metadata = config.output_root / "metadata"
    images.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, Any]] = []
    counts = class_counts(config)
    for clean, count in enumerate(counts):
        for sample_index in range(count):
            difficulty = float(rng.beta(2.0, 2.0))
            flip = bool(rng.random() < _flip_probability(config, clean, difficulty))
            observed = _alternate_label(config, clean, difficulty, rng) if flip else clean
            filename = f"class_{clean:04d}/sample_{sample_index:05d}.png"
            path = images / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            _render_image(config, clean, sample_index, difficulty, rng).save(path, format="PNG", compress_level=9)
            rows.append(
                {
                    "relative_path": filename,
                    "observed_label": f"{observed:04d}",
                    "clean_label": f"{clean:04d}",
                    "is_noisy": int(flip),
                    "difficulty": f"{difficulty:.8f}",
                    "issue": "",
                    "duplicate_of": "",
                }
            )

    eligible = [row for row in rows if not row["issue"]]
    for number in range(config.exact_duplicate_count):
        source = eligible[number % len(eligible)]
        source_path = images / str(source["relative_path"])
        relative = f"duplicates/exact_{number:05d}.png"
        target = images / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)
        rows.append({**source, "relative_path": relative, "issue": "exact_duplicate", "duplicate_of": source["relative_path"]})
    for number in range(config.near_duplicate_count):
        source = eligible[(number + config.exact_duplicate_count) % len(eligible)]
        source_path = images / str(source["relative_path"])
        relative = f"duplicates/near_{number:05d}.png"
        target = images / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_path) as image:
            altered = image.convert("RGB")
            altered.putpixel((number % altered.width, number % altered.height), (0, 0, 0))
            altered.save(target, format="PNG", compress_level=9)
        rows.append({**source, "relative_path": relative, "issue": "near_duplicate", "duplicate_of": source["relative_path"]})
    for number in range(config.corrupt_count):
        clean = number % config.num_classes
        relative = f"invalid/corrupt_{number:05d}.png"
        target = images / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"not-a-valid-image\x00" + bytes([number % 256]))
        rows.append(
            {
                "relative_path": relative,
                "observed_label": f"{clean:04d}",
                "clean_label": f"{clean:04d}",
                "is_noisy": 0,
                "difficulty": "1.00000000",
                "issue": "corrupt",
                "duplicate_of": "",
            }
        )
    for number in range(config.truncated_count):
        source = eligible[(number + config.exact_duplicate_count + config.near_duplicate_count) % len(eligible)]
        source_bytes = (images / str(source["relative_path"])).read_bytes()
        relative = f"invalid/truncated_{number:05d}.png"
        target = images / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_bytes[: max(8, math.floor(len(source_bytes) * 0.65))])
        rows.append({**source, "relative_path": relative, "issue": "truncated", "duplicate_of": source["relative_path"]})

    fieldnames = ["relative_path", "observed_label", "clean_label", "is_noisy", "difficulty", "issue", "duplicate_of"]
    _write_csv(metadata / "samples.csv", fieldnames, rows)
    _write_csv(
        metadata / "classes.csv",
        ["official_class_id", "internal_class_index", "class_name"],
        [
            {
                "official_class_id": f"{index:04d}",
                "internal_class_index": index,
                "class_name": f"synthetic_class_{index:04d}",
            }
            for index in range(config.num_classes)
        ],
    )
    summary = {
        "seed": config.seed,
        "num_classes": config.num_classes,
        "base_class_counts": counts,
        "base_sample_count": sum(counts),
        "total_sample_count": len(rows),
        "noise_mode": config.noise_mode,
        "requested_noise_rate": config.noise_rate,
        "realized_noisy_count": sum(int(row["is_noisy"]) for row in rows if not row["issue"]),
        "exact_duplicate_count": config.exact_duplicate_count,
        "near_duplicate_count": config.near_duplicate_count,
        "corrupt_count": config.corrupt_count,
        "truncated_count": config.truncated_count,
    }
    (metadata / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
