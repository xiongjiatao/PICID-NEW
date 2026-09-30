"""Build a local, hash-recorded TabPFN memory-chunking overlay without editing site-packages."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from picid.research.chunks import atomic_json


def tree_digest(root):
    digest = hashlib.sha256()
    files = (p for p in root.rglob("*") if p.is_file()
             and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"})
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def patch_memory_source(source):
    original = "class MemoryUsageEstimator:\n    SAVE_PEAK_MEM_FACTOR = 8"
    patched = "class MemoryUsageEstimator:\n    SAVE_PEAK_MEM_FACTOR = 16"
    if source.count(original) != 1:
        raise ValueError("Expected exactly one TabPFN 2.2.1 class chunk-factor declaration")
    return source.replace(original, patched, 1)


def build_overlay(source_package, output, lockfile):
    source_package = Path(source_package).resolve()
    output = Path(output).resolve()
    memory_source = source_package / "architectures/base/memory.py"
    if not memory_source.is_file():
        raise FileNotFoundError(memory_source)
    source_hash = tree_digest(source_package)
    version = importlib.metadata.version("tabpfn")
    if version != "2.2.1":
        raise ValueError(f"Expected locked TabPFN 2.2.1, found {version}")
    lockfile = Path(lockfile).resolve()
    lock_text = lockfile.read_text()
    lock_hash = hashlib.sha256(lock_text.encode()).hexdigest()
    patch_file = (
        Path(__file__).resolve().parents[2]
        / "patches/tabpfn/tabpfn_2.2.1_memory_factor16.patch"
    )
    patch_text = patch_file.read_text()
    expected_patch = "-    SAVE_PEAK_MEM_FACTOR = 8\n+    SAVE_PEAK_MEM_FACTOR = 16"
    if expected_patch not in patch_text:
        raise ValueError("Patch file does not describe the expected factor-8 to factor-16 change")
    patch_hash = hashlib.sha256(patch_text.encode()).hexdigest()
    manifest = {
        "patch": patch_file.name,
        "patch_sha256": patch_hash,
        "tabpfn_version": version,
        "uv_lock_sha256": lock_hash,
        "source_package_sha256": source_hash,
        "factor_before": 8,
        "factor_after": 16,
        "scope": "activation chunking only; weights, preprocessing and prediction logic unchanged",
    }
    if output.exists():
        manifest_path = output / "overlay_manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError(f"Existing overlay is not identical to requested build: {output}")
        existing = json.loads(manifest_path.read_text())
        if any(existing.get(key) != value for key, value in manifest.items()):
            raise FileExistsError(f"Existing overlay is not identical to requested build: {output}")
        if tree_digest(output / "tabpfn") != existing.get("patched_package_sha256"):
            raise ValueError(f"Existing overlay content hash mismatch: {output}")
        return manifest_path

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f"{output.name}.tmp-{uuid.uuid4().hex}")
    try:
        package_root = staging / "tabpfn"
        shutil.copytree(source_package, package_root)
        memory_path = package_root / "architectures/base/memory.py"
        memory_path.write_text(patch_memory_source(memory_path.read_text()))
        manifest["patched_package_sha256"] = tree_digest(package_root)
        atomic_json(staging / "overlay_manifest.json", manifest)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "overlay_manifest.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lockfile", type=Path, default=Path("uv.lock"))
    args = parser.parse_args()
    import tabpfn

    path = Path(tabpfn.__file__).resolve().parent
    manifest = build_overlay(path, args.output, args.lockfile)
    print(manifest)
    print(manifest.read_text())


if __name__ == "__main__":
    main()
