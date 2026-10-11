"""Runtime provenance contract for stage-isolated training and inference."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from aictrack1.data.contracts import ALLOWED_COMPETITION_STAGES


_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class CheckpointProvenance:
    competition_stage: str
    class_mapping_sha256: str
    train_manifest_sha256: str
    split_sha256: str
    clip_weight_sha256: str
    num_classes: int

    def __post_init__(self) -> None:
        if self.competition_stage not in ALLOWED_COMPETITION_STAGES:
            raise ValueError(
                f"unsupported checkpoint competition_stage {self.competition_stage!r}"
            )
        for field in (
            "class_mapping_sha256",
            "train_manifest_sha256",
            "split_sha256",
            "clip_weight_sha256",
        ):
            value = getattr(self, field)
            if _SHA256.fullmatch(value) is None:
                raise ValueError(f"checkpoint provenance {field} must be a SHA-256 digest")
        if isinstance(self.num_classes, bool) or self.num_classes < 1:
            raise ValueError("checkpoint provenance num_classes must be positive")

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, context: str = "checkpoint"
    ) -> "CheckpointProvenance":
        required = tuple(cls.__dataclass_fields__)
        missing = [field for field in required if field not in payload]
        if missing:
            raise ValueError(
                f"{context} is missing provenance fields: {', '.join(missing)}"
            )
        if isinstance(payload["num_classes"], bool):
            raise ValueError(f"{context} contains invalid provenance")
        try:
            return cls(
                competition_stage=str(payload["competition_stage"]).lower(),
                class_mapping_sha256=str(payload["class_mapping_sha256"]),
                train_manifest_sha256=str(payload["train_manifest_sha256"]),
                split_sha256=str(payload["split_sha256"]),
                clip_weight_sha256=str(payload["clip_weight_sha256"]),
                num_classes=int(payload["num_classes"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context} contains invalid provenance") from exc


def validate_checkpoint_provenance(
    payload: Mapping[str, Any],
    expected: CheckpointProvenance,
    *,
    context: str = "checkpoint",
) -> None:
    """Reject a checkpoint unless every stage/data/model identity matches."""

    actual = CheckpointProvenance.from_mapping(payload, context=context)
    for field in CheckpointProvenance.__dataclass_fields__:
        actual_value = getattr(actual, field)
        expected_value = getattr(expected, field)
        if actual_value != expected_value:
            raise ValueError(
                f"{context} {field} mismatch: {actual_value!r} != {expected_value!r}"
            )
