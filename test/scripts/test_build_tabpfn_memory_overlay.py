import json
from pathlib import Path

import pytest

from scripts.research.build_tabpfn_memory_overlay import build_overlay, patch_memory_source


def test_patch_changes_only_the_class_memory_chunk_factor():
    source = (
        "SAVE_PEAK_MEM_FACTOR = 8\n\n"
        "class MemoryUsageEstimator:\n"
        "    SAVE_PEAK_MEM_FACTOR = 8\n"
    )
    result = patch_memory_source(source)
    assert result == (
        "SAVE_PEAK_MEM_FACTOR = 8\n\n"
        "class MemoryUsageEstimator:\n"
        "    SAVE_PEAK_MEM_FACTOR = 16\n"
    )


def test_patch_fails_closed_for_unrecognized_upstream_code():
    with pytest.raises(ValueError, match="exactly one"):
        patch_memory_source("class MemoryUsageEstimator:\n    SAVE_PEAK_MEM_FACTOR = 4\n")


def test_overlay_build_records_upstream_and_patched_hashes(tmp_path):
    source = tmp_path / "source" / "tabpfn"
    memory = source / "architectures/base/memory.py"
    memory.parent.mkdir(parents=True)
    memory.write_text("class MemoryUsageEstimator:\n    SAVE_PEAK_MEM_FACTOR = 8\n")
    lockfile = Path(__file__).resolve().parents[2] / "uv.lock"

    manifest_path = build_overlay(source, tmp_path / "overlay", lockfile)
    manifest = json.loads(manifest_path.read_text())
    patched = (tmp_path / "overlay/tabpfn/architectures/base/memory.py").read_text()
    assert "SAVE_PEAK_MEM_FACTOR = 16" in patched
    assert manifest["factor_before"] == 8
    assert manifest["factor_after"] == 16
    assert manifest["source_package_sha256"]
    assert manifest["patched_package_sha256"]
    assert build_overlay(source, tmp_path / "overlay", lockfile) == manifest_path
