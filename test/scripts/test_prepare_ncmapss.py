"""Verify selective extraction, source-name mapping and overwrite protection."""

import importlib.util
from io import BytesIO
import json
from pathlib import Path
import sys
import zipfile

import pytest

spec = importlib.util.spec_from_file_location(
    "prepare_ncmapss", Path(__file__).parents[2] / "scripts/research/prepare_ncmapss.py"
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def fixture_archive(tmp_path, monkeypatch, nested=False):
    archive = tmp_path / "NASA_N-CMAPSS.zip"
    members = BytesIO()
    with zipfile.ZipFile(members, "w") as zipped:
        for name in (
            "N-CMAPSS_DS01-005.h5",
            "N-CMAPSS_DS04.h5",
            "N-CMAPSS_DS05.h5",
            "N-CMAPSS_DS07.h5",
        ):
            zipped.writestr("data/" + name, b"fixture")
        zipped.writestr("../../ignored.txt", b"must not extract")
    with zipfile.ZipFile(archive, "w") as zipped:
        if nested:
            zipped.writestr("17. Turbofan/data_set.zip", members.getvalue())
        else:
            zipped.writestr("data_set.zip", members.getvalue())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--data-root",
            str(tmp_path),
            "--status",
            str(tmp_path / "status.json"),
            "--archive",
            str(archive),
        ],
    )


def test_extracts_exact_sources_and_records_original_names(tmp_path, monkeypatch):
    fixture_archive(tmp_path, monkeypatch, nested=True)
    prepare.main()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "DATA_READY_PROTOCOL_AUDIT_STILL_REQUIRED"
    assert len(status["files"]) == 4
    assert (tmp_path / "N-CMAPSS" / "N-CMAPSS_DS01.h5").read_bytes() == b"fixture"
    assert not (tmp_path / "ignored.txt").exists()
    assert status["nested_archive_member"] == "17. Turbofan/data_set.zip"
    assert all(
        item["archive_member"].startswith("17. Turbofan/data_set.zip!/")
        for item in status["files"]
    )


def test_existing_dataset_is_preserved(tmp_path, monkeypatch):
    fixture_archive(tmp_path, monkeypatch)
    directory = tmp_path / "N-CMAPSS"
    directory.mkdir()
    existing = directory / "N-CMAPSS_DS01.h5"
    existing.write_bytes(b"user data")
    with pytest.raises(RuntimeError, match="overwrite"):
        prepare.main()
    assert existing.read_bytes() == b"user data"
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "FAILED"
