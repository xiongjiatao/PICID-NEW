"""Re-key a valid failed-run XJTU feature boundary after a serializer-only fix.

This utility does not recompute or alter feature values. It takes the complete
``DatasetContainer`` saved after ``final_scaler_features`` by a run that failed
while pickling transform metadata, pairs it with freshly instantiated (now
pickleable) transform objects and the datasource metadata from the successful
load/split cache, and writes a new boundary cache under the current source hash.

The old failed cache is read-only and retained for provenance. The new cache is
written only if its key does not already exist. The intended use is to avoid
repeating the expensive XJTU time-domain statistic pass after a cache metadata
serialization failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import joblib
from omegaconf import OmegaConf

from picid.data.cache.offline import FileSystemCache
from picid.transforms.base.transform_manager import ConfigTransformManager
from picid.utils.hash_utils import compute_cache_key

BOUNDARY_NAME = "final_scaler_features"
EXPERIMENT = "xjtu_sy/prognostics/phmd_split/combined/lstm"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--failed-boundary-key", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    run_dir = args.run_dir.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    failed_dir = cache_root / "boundary" / args.failed_boundary_key
    pending_path = failed_dir / "meta_pending.json"
    source_data_path = failed_dir / "data.pkl"
    loaded_metadata_path = cache_root / "loaded_and_splitted_data" / "metadata.pkl"
    if not pending_path.is_file() or not source_data_path.is_file():
        raise FileNotFoundError(
            "Expected a failed boundary cache with meta_pending.json and data.pkl: "
            f"{failed_dir}"
        )
    if not loaded_metadata_path.is_file():
        raise FileNotFoundError(f"Missing verified datasource metadata cache: {loaded_metadata_path}")
    if (failed_dir / "meta.json").exists() or (failed_dir / "hash.txt").exists():
        raise RuntimeError(
            "The supplied boundary is already committed/valid; refusing recovery."
        )

    with pending_path.open(encoding="utf-8") as stream:
        boundary_config = json.load(stream)
    transform_cfg = boundary_config.get("transforms", {})
    if BOUNDARY_NAME not in transform_cfg:
        raise ValueError(
            f"Failed cache is not at expected boundary {BOUNDARY_NAME!r}."
        )

    # Importing the main runner registers the project-specific OmegaConf
    # resolvers required by the saved Hydra configuration. It does not execute
    # training or initialize CUDA.
    import picid.run  # noqa: F401

    saved_config = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    overrides_path = run_dir / ".hydra" / "overrides.yaml"
    run_overrides = list(OmegaConf.load(overrides_path))
    if f"experiment={EXPERIMENT}" not in run_overrides or "seed=72" not in run_overrides:
        raise ValueError(
            "Run configuration does not match the expected XJTU PHMD seed-72 LSTM."
        )
    if saved_config.datasource.split_mode != "phmd_split":
        raise ValueError("Saved run does not use the required XJTU PHMD split.")
    if saved_config.datasource.download_policy != "local_only":
        raise ValueError("Saved run was not constrained to local-only PHMD data.")

    manager = ConfigTransformManager(transforms_config=saved_config.transforms)
    if BOUNDARY_NAME not in manager.get_cache_point_names():
        raise ValueError(f"{BOUNDARY_NAME!r} is not an active cache point.")
    all_transforms = manager.get_transforms()
    names_after_boundary = set(manager.get_transform_names_after(BOUNDARY_NAME))
    prefix_transforms = {
        name: transform
        for name, transform in all_transforms.items()
        if name not in names_after_boundary
    }

    # The feature payload was fully written before metadata serialization
    # failed. Validate it in isolation; never trust or unpickle the partial
    # boundary metadata from the failed save.
    cached_dataset = joblib.load(source_data_path)
    cached_dataset.validate()
    source_dataset_fingerprint = hashlib.sha256(
        pickle.dumps(cached_dataset, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()
    datasource_metadata = joblib.load(loaded_metadata_path)
    metadata = {"transforms": prefix_transforms, "meta_data": datasource_metadata}
    pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL)

    library_dirs = [
        str(root / "picid" / "data" / "datasources"),
        str(root / "picid" / "transforms"),
    ]
    cache_key = compute_cache_key(
        boundary_config, library_dirs, extensions=[".py"]
    )
    if cache_key == args.failed_boundary_key:
        raise RuntimeError(
            "Source hash did not change; cache recovery would reproduce the failed key."
        )
    target_dir = cache_root / "boundary" / cache_key
    if target_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing boundary cache: {target_dir}"
        )

    cache = FileSystemCache()
    cache.write_meta(
        cache_dir=str(cache_root),
        stage="boundary",
        config=boundary_config,
        library_dir=library_dirs,
        extensions=[".py"],
    )
    cache.save(
        cache_dir=str(cache_root),
        stage="boundary",
        data=cached_dataset,
        metadata=metadata,
        config=boundary_config,
        library_dir=library_dirs,
        extensions=[".py"],
    )
    # Verify the complete metadata artifact and canonical cache key before
    # reporting success; data.pkl is identical in value, not copied from raw data.
    _, stored_hash = cache.load_metadata(
        str(cache_root), "boundary", cache_key=cache_key
    )
    if stored_hash != cache_key:
        raise RuntimeError("Recovered boundary cache hash verification failed.")
    recovered_metadata = joblib.load(target_dir / "metadata.pkl")
    if not isinstance(recovered_metadata, dict) or set(recovered_metadata) != {
        "transforms",
        "meta_data",
    }:
        raise RuntimeError("Recovered boundary metadata has an unexpected shape.")
    recovered_dataset = joblib.load(target_dir / "data.pkl")
    recovered_dataset.validate()
    recovered_dataset_fingerprint = hashlib.sha256(
        pickle.dumps(recovered_dataset, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()
    if source_dataset_fingerprint != recovered_dataset_fingerprint:
        raise RuntimeError(
            "Recovered boundary data payload differs from the complete source payload."
        )

    report = {
        "status": "RECOVERED_BOUNDARY_CACHE",
        "run_dir": str(run_dir),
        "failed_boundary_key": args.failed_boundary_key,
        "new_boundary_key": cache_key,
        "boundary_name": BOUNDARY_NAME,
        "source_data_path": str(source_data_path),
        "source_data_sha256": sha256_file(source_data_path),
        "source_dataset_fingerprint": source_dataset_fingerprint,
        "recovered_data_path": str(target_dir / "data.pkl"),
        "recovered_data_sha256": sha256_file(target_dir / "data.pkl"),
        "recovered_dataset_fingerprint": recovered_dataset_fingerprint,
        "unit_names": datasource_metadata.get("unit_names"),
        "pre_boundary_transform_count": len(prefix_transforms),
        "source_tree": str(root),
        "note": "Feature values were not recomputed or modified; source and recovered DatasetContainer logical fingerprints match exactly.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
