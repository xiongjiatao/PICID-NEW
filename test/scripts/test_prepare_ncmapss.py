"""Verify selective extraction, source-name mapping and overwrite protection."""

import importlib.util
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


def fixture_archive(tmp_path, monkeypatch):
    archive = tmp_path / "NASA_N-CMAPSS.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for name in (
            "N-CMAPSS_DS01-005.h5",
            "N-CMAPSS_DS04.h5",
            "N-CMAPSS_DS05.h5",
            "N-CMAPSS_DS07.h5",
        ):
            zipped.writestr("data/" + name, b"fixture")
        zipped.writestr("../../ignored.txt", b"must not extract")

    class Response:
        headers = {"Content-Length": str(archive.stat().st_size), "ETag": "fixture"}

        def raise_for_status(self):
            pass

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def head(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(prepare.requests, "Session", Session)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--data-root",
            str(tmp_path),
            "--status",
            str(tmp_path / "status.json"),
        ],
    )


def test_extracts_exact_sources_and_records_original_names(tmp_path, monkeypatch):
    fixture_archive(tmp_path, monkeypatch)
    prepare.main()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "DATA_READY_PROTOCOL_AUDIT_STILL_REQUIRED"
    assert len(status["files"]) == 4
    assert (tmp_path / "N-CMAPSS" / "N-CMAPSS_DS01.h5").read_bytes() == b"fixture"
    assert not (tmp_path / "ignored.txt").exists()


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
