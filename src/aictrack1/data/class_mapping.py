"""Lossless mapping between official string IDs and internal class indices."""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClassEntry:
    official_class_id: str
    internal_class_index: int
    class_name: str = ""


class ClassMapping:
    def __init__(self, entries: list[ClassEntry]) -> None:
        if not entries:
            raise ValueError("class mapping cannot be empty")
        official = [entry.official_class_id for entry in entries]
        internal = [entry.internal_class_index for entry in entries]
        if any(not value for value in official):
            raise ValueError("official class IDs must be non-empty strings")
        if len(set(official)) != len(official):
            raise ValueError("duplicate official class ID")
        if len(set(internal)) != len(internal):
            raise ValueError("duplicate internal class index")
        if sorted(internal) != list(range(len(entries))):
            raise ValueError("internal class indices must be contiguous from zero")
        self.entries = tuple(sorted(entries, key=lambda entry: entry.internal_class_index))
        self._to_internal = {entry.official_class_id: entry.internal_class_index for entry in self.entries}
        self._to_official = {entry.internal_class_index: entry.official_class_id for entry in self.entries}

    @classmethod
    def from_csv(cls, path: str | Path) -> "ClassMapping":
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"official_class_id", "internal_class_index"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"class mapping requires columns {sorted(required)}")
            entries = [
                ClassEntry(
                    official_class_id=str(row["official_class_id"]),
                    internal_class_index=int(row["internal_class_index"]),
                    class_name=str(row.get("class_name", "") or ""),
                )
                for row in reader
            ]
        return cls(entries)

    def to_internal(self, official_class_id: str) -> int:
        try:
            return self._to_internal[str(official_class_id)]
        except KeyError as exc:
            raise KeyError(f"unknown official class ID {official_class_id!r}") from exc

    def to_official(self, internal_class_index: int) -> str:
        try:
            return self._to_official[int(internal_class_index)]
        except KeyError as exc:
            raise KeyError(f"unknown internal class index {internal_class_index!r}") from exc

    def digest(self) -> str:
        payload = "\n".join(
            f"{entry.internal_class_index}\t{entry.official_class_id}\t{entry.class_name}"
            for entry in self.entries
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_train_directories(cls, root: str | Path) -> "ClassMapping":
        directory = Path(root)
        labels = sorted(path.name for path in directory.iterdir() if path.is_dir())
        if not labels:
            raise ValueError("training directory contains no class subdirectories")
        invalid = [label for label in labels if re.fullmatch(r"\d{4}", label) is None]
        if invalid:
            raise ValueError(f"class directories must be four digits; examples: {invalid[:3]}")
        return cls([ClassEntry(label, index) for index, label in enumerate(labels)])

    def write_csv(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["official_class_id", "internal_class_index", "class_name"],
                lineterminator="\n",
            )
            writer.writeheader()
            for entry in self.entries:
                writer.writerow(
                    {
                        "official_class_id": entry.official_class_id,
                        "internal_class_index": entry.internal_class_index,
                        "class_name": entry.class_name,
                    }
                )

    def __len__(self) -> int:
        return len(self.entries)
