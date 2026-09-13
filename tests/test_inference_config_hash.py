from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.infer_and_build_submission import (
    SUPPORTED_FINAL_PEFT_MODES,
    _text_sha256_variants,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_text_sha256_variants_accepts_only_line_ending_changes(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_bytes(b"experiment:\n  seed: 1\n")

    variants = _text_sha256_variants(config)

    assert _sha256(b"experiment:\n  seed: 1\n") in variants
    assert _sha256(b"experiment:\r\n  seed: 1\r\n") in variants
    assert _sha256(b"experiment:\n  seed: 2\n") not in variants


def test_final_inference_accepts_both_valid_lora_scopes() -> None:
    assert SUPPORTED_FINAL_PEFT_MODES == {
        "lora_last_attention",
        "lora_last_n_attention",
    }
