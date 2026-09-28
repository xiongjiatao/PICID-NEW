"""Atomic, fingerprint-checked prediction chunks with an ordered result contract."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .protocol import digest


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
    os.replace(tmp, path)


def predict_chunks(predict, X, directory, identity, batch_size, member="all"):
    """One writer per directory; caller must include fitted-model and code digests.

    An execution protocol must pass equivalence testing before using this routine
    for formal evaluation. A fingerprint match does not establish equivalence.
    """
    required = {"config", "weights", "training_input", "code", "physical_gpus"}
    if not required <= identity.keys() or batch_size < 1 or len(X) == 0:
        raise ValueError("Missing prediction identity, invalid batch size or empty input")
    X = np.ascontiguousarray(X)
    fingerprint = digest({**identity, "query_hash": hashlib.sha256(X.tobytes()).hexdigest(),
                          "shape": X.shape, "dtype": str(X.dtype),
                          "batch_size": batch_size, "member": member})
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".writer.lock"
    fd = os.open(lock, os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise
    try:
        manifest = directory / "identity.json"
        if manifest.exists():
            if json.loads(manifest.read_text())["fingerprint"] != fingerprint:
                raise ValueError("Resume fingerprint mismatch")
        else:
            atomic_json(manifest, {"fingerprint": fingerprint, "identity": identity})
        outputs = []
        started = time.monotonic()
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            path = directory / f"{start:012d}_{end:012d}.npy"
            meta = path.with_suffix(".json")
            if path.exists() and meta.exists():
                if file_digest(path) != json.loads(meta.read_text())["sha256"]:
                    raise ValueError("Corrupt prediction chunk")
                result = np.load(path, allow_pickle=False)
            else:
                result = np.asarray(predict(X[start:end]))
                if len(result) != end - start or not np.isfinite(result).all():
                    raise ValueError("Invalid prediction rows")
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    np.save(stream, result, allow_pickle=False)
                os.replace(temporary, path)
                atomic_json(meta, {"start": start, "end": end, "sha256": file_digest(path)})
            if len(result) != end - start or not np.isfinite(result).all():
                raise ValueError("Invalid restored prediction rows")
            outputs.append(result)
            elapsed = time.monotonic() - started
            atomic_json(directory / "progress.json", {
                "completed": end, "total": len(X), "member": member,
                "elapsed_seconds": elapsed, "status": "complete" if end == len(X) else "running",
                "rows_per_second_including_restore": end / max(elapsed, 1e-9),
                "eta_seconds_including_restore": elapsed * (len(X) - end) / end})
        return np.concatenate(outputs)
    finally:
        os.close(fd)
        # Keep the inode: flock releases automatically even after process death.
