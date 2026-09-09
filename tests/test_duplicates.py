from __future__ import annotations

from aictrack1.data.duplicates import assign_duplicate_groups, hamming_distance
from aictrack1.data.manifest import SampleRecord


def test_hamming_and_grouping() -> None:
    assert hamming_distance("0000000000000000", "0000000000000003") == 2
    records = [
        SampleRecord("a", "a.png", "train", "0000", 0, "0000", sha256="same", dhash="0000000000000000", decode_status="ok"),
        SampleRecord("b", "b.png", "train", "0000", 0, "0000", sha256="same", dhash="0000000000000001", decode_status="ok"),
        SampleRecord("c", "c.png", "train", "0001", 1, "0001", sha256="other", dhash="ffffffffffffffff", decode_status="ok"),
    ]
    grouped, diagnostics = assign_duplicate_groups(records, near_hamming=1, max_bucket_size=10)
    assert grouped[0].duplicate_group == grouped[1].duplicate_group
    assert grouped[0].near_duplicate_group == grouped[1].near_duplicate_group
    assert diagnostics["exact_groups"] == 1


def test_identical_perceptual_hashes_survive_oversized_lsh_bucket() -> None:
    records = [
        SampleRecord(str(index), f"{index}.png", "train", "0000", 0, "0000", sha256=str(index), dhash="1111111111111111", decode_status="ok")
        for index in range(4)
    ]
    grouped, diagnostics = assign_duplicate_groups(records, near_hamming=1, max_bucket_size=2)
    assert len({record.near_duplicate_group for record in grouped}) == 1
    assert grouped[0].near_duplicate_group
    assert diagnostics["near_skipped_buckets"] == 4
