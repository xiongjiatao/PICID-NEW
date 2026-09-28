"""Copy the pinned dependency and apply a hash-guarded patch outside .venv."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from picid.research.chunks import file_digest  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    metadata = json.loads((root / "patches/tabpfn-2.2.1-bounded-nan.json").read_text())
    source = Path(importlib.util.find_spec("tabpfn").origin).parent
    relative = Path(metadata["file"]).relative_to("tabpfn")
    if file_digest(source / relative) != metadata["source_sha256"]:
        raise RuntimeError("Dependency source differs from audited patch base")
    args.destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, args.destination / "tabpfn", ignore=shutil.ignore_patterns("__pycache__"))
    subprocess.run(["patch", "-p1", "-i", str(root / "patches/tabpfn-2.2.1-bounded-nan.patch")],
                   cwd=args.destination, check=True)
    assert file_digest(args.destination / metadata["file"]) == metadata["patched_sha256"]
    print(args.destination.resolve())
