"""Deterministic exact and near-duplicate grouping without all-pairs comparison."""

from __future__ import annotations

from collections import defaultdict

from .manifest import SampleRecord, replace_groups


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _number_groups(groups: dict[str, list[int]], prefix: str) -> dict[int, str]:
    result: dict[int, str] = {}
    ordered = sorted(
        (members for members in groups.values() if len(members) > 1),
        key=lambda members: tuple(members),
    )
    for number, members in enumerate(ordered, start=1):
        group_id = f"{prefix}-{number:06d}"
        for index in members:
            result[index] = group_id
    return result


def assign_duplicate_groups(
    records: list[SampleRecord],
    *,
    near_hamming: int = 3,
    max_bucket_size: int = 500,
) -> tuple[list[SampleRecord], dict[str, int]]:
    if not 0 <= near_hamming <= 3:
        raise ValueError("near_hamming must be between 0 and 3")
    if max_bucket_size < 2:
        raise ValueError("max_bucket_size must be at least 2")

    exact_raw: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record.sha256:
            exact_raw[record.sha256].append(index)
    exact_ids = _number_groups(exact_raw, "exact")

    valid = [(index, record.dhash) for index, record in enumerate(records) if record.dhash]
    union = UnionFind(len(records))
    identical_hashes: dict[str, list[int]] = defaultdict(list)
    for index, dhash in valid:
        identical_hashes[dhash].append(index)
    for members in identical_hashes.values():
        for index in members[1:]:
            union.union(members[0], index)
    buckets: dict[tuple[int, str], list[int]] = defaultdict(list)
    skipped_buckets = 0
    compared_pairs: set[tuple[int, int]] = set()
    for index, dhash in valid:
        for band in range(4):
            buckets[(band, dhash[band * 4 : (band + 1) * 4])].append(index)
    for members in buckets.values():
        if len(members) > max_bucket_size:
            skipped_buckets += 1
            continue
        for offset, left in enumerate(members):
            for right in members[offset + 1 :]:
                pair = (left, right) if left < right else (right, left)
                if pair in compared_pairs:
                    continue
                compared_pairs.add(pair)
                if hamming_distance(records[left].dhash, records[right].dhash) <= near_hamming:
                    union.union(left, right)

    near_raw: dict[str, list[int]] = defaultdict(list)
    for index, _ in valid:
        near_raw[str(union.find(index))].append(index)
    near_ids = _number_groups(near_raw, "near")
    updated = [replace_groups(record, exact_ids.get(i, ""), near_ids.get(i, "")) for i, record in enumerate(records)]
    diagnostics = {
        "exact_groups": len(set(exact_ids.values())),
        "near_groups": len(set(near_ids.values())),
        "near_candidate_pairs": len(compared_pairs),
        "near_skipped_buckets": skipped_buckets,
    }
    return updated, diagnostics
