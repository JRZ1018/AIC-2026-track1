from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pytest

from aictrack1.data.class_mapping import ClassEntry, ClassMapping
from aictrack1.submission import (
    SubmissionSpec,
    build_submission_package,
    submission_keys_from_relative_paths,
    validate_submission_csv,
    validate_submission_zip,
)


EXPECTED = ["B.JPG", "a.jpg", "c.jpg"]
METADATA = {
    "run_id": "mock-run",
    "checkpoint_sha256": "1" * 64,
    "config_hash": "2" * 64,
    "data_hash": "3" * 64,
    "git_commit": "4" * 40,
    "inference_command": "python scripts/infer.py --config mock.yaml",
}


def _mapping() -> ClassMapping:
    return ClassMapping(
        [ClassEntry("0001", 0), ClassEntry("0123", 1), ClassEntry("0456", 2)]
    )


def _build(tmp_path: Path, *, spec: SubmissionSpec = SubmissionSpec()):
    predictions = {"a.jpg": 1, "B.JPG": 0, "c.jpg": 2}
    return build_submission_package(
        predictions,
        EXPECTED,
        _mapping(),
        tmp_path / "submission",
        run_metadata=METADATA,
        spec=spec,
    )


def test_package_is_exact_order_no_header_and_one_file_zip(tmp_path: Path) -> None:
    package = _build(tmp_path)
    assert package.csv_path.read_text(encoding="utf-8") == (
        "B.JPG,0001\na.jpg,0123\nc.jpg,0456\n"
    )
    with zipfile.ZipFile(package.zip_path) as archive:
        assert archive.namelist() == ["pred_results.csv"]
        assert archive.read("pred_results.csv") == package.csv_path.read_bytes()
    manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["record_count"] == 3
    assert manifest["run_metadata"] == METADATA
    assert manifest["csv_sha256"] == package.csv_sha256
    assert manifest["zip_sha256"] == package.zip_sha256


def test_explicit_header_mode_round_trips(tmp_path: Path) -> None:
    spec = SubmissionSpec(include_header=True)
    package = _build(tmp_path, spec=spec)
    assert package.csv_path.read_text(encoding="utf-8").startswith("filename,label\n")
    validate_submission_zip(
        package.zip_path,
        EXPECTED,
        ["0001", "0123", "0456"],
        spec=spec,
    )


@pytest.mark.parametrize(
    "predictions,match",
    [
        ({"B.JPG": 0, "a.jpg": 1}, "mismatch"),
        ({"B.JPG": 0, "a.jpg": 1, "c.jpg": 2, "extra.jpg": 0}, "mismatch"),
        ({"B.JPG": 0, "a.jpg": 1, "c.jpg": 99}, "unknown class index"),
    ],
)
def test_builder_rejects_incomplete_extra_or_unknown_predictions(
    tmp_path: Path, predictions: dict[str, int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        build_submission_package(
            predictions,
            EXPECTED,
            _mapping(),
            tmp_path,
            run_metadata=METADATA,
        )


def test_builder_refuses_overwrite_and_incomplete_metadata(tmp_path: Path) -> None:
    _build(tmp_path)
    with pytest.raises(FileExistsError, match="overwrite"):
        _build(tmp_path)
    bad_metadata = dict(METADATA)
    bad_metadata.pop("data_hash")
    with pytest.raises(ValueError, match="missing required"):
        build_submission_package(
            {"B.JPG": 0, "a.jpg": 1, "c.jpg": 2},
            EXPECTED,
            _mapping(),
            tmp_path / "other",
            run_metadata=bad_metadata,
        )


@pytest.mark.parametrize(
    "contents,match",
    [
        ("filename,label\nB.JPG,0001\na.jpg,0123\nc.jpg,0456\n", "header"),
        ("a.jpg,0123\nB.JPG,0001\nc.jpg,0456\n", "order"),
        ("B.JPG,0001\na.jpg,0123\na.jpg,0456\n", "duplicate"),
        ("B.JPG,0001\na.jpg,9999\nc.jpg,0456\n", "invalid official"),
        ("B.JPG,0001,extra\na.jpg,0123\nc.jpg,0456\n", "exactly two"),
    ],
)
def test_csv_validator_rejects_invalid_files(
    tmp_path: Path, contents: str, match: str
) -> None:
    path = tmp_path / "pred_results.csv"
    path.write_text(contents, encoding="utf-8", newline="")
    with pytest.raises(ValueError, match=match):
        validate_submission_csv(path, EXPECTED, ["0001", "0123", "0456"])


def test_csv_validator_rejects_bom_and_crlf(tmp_path: Path) -> None:
    path = tmp_path / "pred_results.csv"
    path.write_bytes(b"\xef\xbb\xbfB.JPG,0001\n")
    with pytest.raises(ValueError, match="BOM"):
        validate_submission_csv(path, ["B.JPG"], ["0001"])
    path.write_bytes(b"B.JPG,0001\r\n")
    with pytest.raises(ValueError, match="LF"):
        validate_submission_csv(path, ["B.JPG"], ["0001"])


def test_zip_validator_rejects_extra_member(tmp_path: Path) -> None:
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("pred_results.csv", "B.JPG,0001\n")
        archive.writestr("notes.txt", "not allowed")
    with pytest.raises(ValueError, match="only"):
        validate_submission_zip(path, ["B.JPG"], ["0001"])


def test_writer_uses_csv_quoting_but_validator_preserves_exact_filename(tmp_path: Path) -> None:
    mapping = _mapping()
    expected = ["comma,name.jpg"]
    package = build_submission_package(
        {"comma,name.jpg": 0},
        expected,
        mapping,
        tmp_path,
        run_metadata=METADATA,
    )
    with package.csv_path.open(encoding="utf-8", newline="") as handle:
        assert list(csv.reader(handle)) == [["comma,name.jpg", "0001"]]


def test_non_numeric_official_ids_are_validated_by_mapping_membership(
    tmp_path: Path,
) -> None:
    mapping = ClassMapping(
        [
            ClassEntry("cls_001", 0),
            ClassEntry("物种 甲", 1),
            ClassEntry("10001", 2),
        ]
    )
    expected = ["a.jpg", "b.jpg", "c.jpg"]
    package = build_submission_package(
        {"a.jpg": 0, "b.jpg": 1, "c.jpg": 2},
        expected,
        mapping,
        tmp_path,
        run_metadata=METADATA,
    )
    with package.csv_path.open(encoding="utf-8", newline="") as handle:
        assert list(csv.reader(handle)) == [
            ["a.jpg", "cls_001"],
            ["b.jpg", "物种 甲"],
            ["c.jpg", "10001"],
        ]


def test_explicit_official_id_pattern_is_optional_and_enforced(tmp_path: Path) -> None:
    mapping = ClassMapping([ClassEntry("cls_001", 0)])
    spec = SubmissionSpec(official_class_id_pattern=r"\d{4}")
    with pytest.raises(ValueError, match="configured pattern"):
        build_submission_package(
            {"a.jpg": 0},
            ["a.jpg"],
            mapping,
            tmp_path,
            run_metadata=METADATA,
            spec=spec,
        )


def test_nested_test_paths_support_explicit_relative_path_mode(tmp_path: Path) -> None:
    spec = SubmissionSpec(filename_mode="relative_path")
    relative_paths = ["folder-a/x.jpg", "folder-b/x.jpg", "root.jpg"]
    keys = submission_keys_from_relative_paths(relative_paths, spec=spec)
    assert keys == tuple(relative_paths)
    package = build_submission_package(
        {"folder-a/x.jpg": 0, "folder-b/x.jpg": 1, "root.jpg": 2},
        keys,
        _mapping(),
        tmp_path,
        run_metadata=METADATA,
        spec=spec,
    )
    assert package.csv_path.read_text(encoding="utf-8") == (
        "folder-a/x.jpg,0001\nfolder-b/x.jpg,0123\nroot.jpg,0456\n"
    )


def test_basename_mode_rejects_nested_path_collisions() -> None:
    with pytest.raises(ValueError, match="collide"):
        submission_keys_from_relative_paths(
            ["folder-a/x.jpg", "folder-b/X.JPG"],
            spec=SubmissionSpec(filename_mode="basename"),
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "../escape.jpg",
        "folder/../escape.jpg",
        "/absolute.jpg",
        "folder//x.jpg",
        "folder/bad\tname.jpg",
    ],
)
def test_submission_keys_reject_unsafe_relative_paths(relative_path: str) -> None:
    with pytest.raises(ValueError, match="unsafe|control characters"):
        submission_keys_from_relative_paths(
            [relative_path],
            spec=SubmissionSpec(filename_mode="relative_path"),
        )
