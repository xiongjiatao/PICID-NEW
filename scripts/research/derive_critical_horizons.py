"""Derive fixed warning horizons from training devices only."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import h5py
import numpy as np


SOURCES = (1, 4, 5, 7)
TRAIN_UNITS = (1, 2, 3, 4, 5)
FRACTIONS = (0.05, 0.10, 0.20)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_names(values) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def read_training_device_max_rul(path: Path, source: int, train_units=TRAIN_UNITS):
    """Read per-device maximum raw RUL from the development split.

    N-CMAPSS dev data contain train units 1--5 and validation unit 6 under the
    registered source split. This function reads only the training unit IDs.
    """
    with h5py.File(path, "r") as data:
        names = _decode_names(data["A_var"][:])
        if "unit" not in names:
            raise ValueError(f"DS{source:02d} has no unit column")
        unit_ids = np.asarray(data["A_dev"][:, names.index("unit")]).astype(np.int64)
        raw_rul = np.asarray(data["Y_dev"][:, 0], dtype=np.float64)
    if len(unit_ids) != len(raw_rul) or not np.isfinite(raw_rul).all():
        raise ValueError(f"DS{source:02d} has unaligned or non-finite dev targets")
    observed = set(int(value) for value in np.unique(unit_ids))
    missing = sorted(set(train_units) - observed)
    if missing:
        raise ValueError(f"DS{source:02d} is missing training units {missing}")
    result = {}
    for unit in train_units:
        mask = unit_ids == unit
        if not mask.any():
            raise ValueError(f"DS{source:02d}-unit{unit:02d} has no dev rows")
        result[f"DS{source:02d}-unit{unit:02d}"] = float(np.max(raw_rul[mask]))
    return result


def derive_training_horizons(data_root: Path, sources=SOURCES, train_units=TRAIN_UNITS,
                             fractions=FRACTIONS):
    lifetimes = {}
    hashes = {}
    for source in sources:
        path = data_root / f"N-CMAPSS_DS{source:02d}.h5"
        if not path.is_file():
            raise FileNotFoundError(path)
        lifetimes.update(read_training_device_max_rul(path, source, train_units))
        hashes[path.name] = sha256_file(path)
    values = list(lifetimes.values())
    median = float(statistics.median(values))
    if median <= 0:
        raise ValueError("Training-device median maximum RUL must be positive")
    return {
        "protocol": "training_only_max_rul_fraction",
        "dataset": "nc_p",
        "raw_rul_unit": "N-CMAPSS Y_dev target unit",
        "source_ids": list(sources),
        "training_units_by_source": list(train_units),
        "validation_unit_by_source": 6,
        "training_device_count": len(lifetimes),
        "training_device_max_rul": lifetimes,
        "median_training_device_max_rul": median,
        "horizons": {
            f"{int(fraction * 100)}pct": {
                "fraction": fraction,
                "rul_threshold": median * fraction,
            }
            for fraction in fractions
        },
        "raw_data_sha256": hashes,
    }


def derive_xjtu_training_horizons(data_root: Path, training_bearings=None,
                                  fractions=FRACTIONS):
    """Derive XJTU horizons from PHMD training-bearing acquisition indices only.

    Each CSV is one acquisition. PHMD labels the first acquisition with RUL
    ``N-1`` and the last with zero, so the training lifetime used here is
    ``N-1`` in acquisition intervals. Evaluation-bearing directories are never
    read by this function.
    """
    if training_bearings is None:
        from picid.data.datasources.phmd_xjtu_sy import XJTU_SY_SPLIT_ASSIGNMENTS_BY_MODE

        training_bearings = XJTU_SY_SPLIT_ASSIGNMENTS_BY_MODE["phmd_split"]["train"]
    training_bearings = tuple(training_bearings)
    lifetimes = {}
    index_hashes = {}
    acquisition_counts = {}
    for bearing in training_bearings:
        folder = data_root / "train" / f"Bearing{bearing}"
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        indices = []
        for path in folder.glob("*.csv"):
            try:
                indices.append(int(path.stem))
            except ValueError as exc:
                raise ValueError(f"Non-numeric acquisition file in {folder}: {path.name}") from exc
        indices.sort()
        expected = list(range(1, len(indices) + 1))
        if not indices or indices != expected:
            raise ValueError(f"{folder} acquisition indices are empty, duplicated, or gapped")
        acquisition_counts[f"Bearing{bearing}"] = len(indices)
        lifetimes[f"Bearing{bearing}"] = float(len(indices) - 1)
        listing = "\n".join(f"{index}.csv" for index in indices).encode()
        index_hashes[f"Bearing{bearing}"] = hashlib.sha256(listing).hexdigest()
    values = list(lifetimes.values())
    median = float(statistics.median(values))
    if median <= 0:
        raise ValueError("Training-bearing median maximum RUL must be positive")
    return {
        "protocol": "training_only_max_rul_fraction",
        "dataset": "xjtu",
        "raw_rul_unit": "PHMD bearing acquisition intervals",
        "lifetime_definition": "N ordered acquisition files; maximum PHMD RUL is N-1",
        "split": "phmd_split",
        "training_bearings": list(training_bearings),
        "training_device_count": len(lifetimes),
        "training_device_acquisition_count": acquisition_counts,
        "training_device_max_rul": lifetimes,
        "median_training_device_max_rul": median,
        "horizons": {
            f"{int(fraction * 100)}pct": {
                "fraction": fraction,
                "rul_threshold": median * fraction,
            }
            for fraction in fractions
        },
        "acquisition_index_listing_sha256": index_hashes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("nc_p", "xjtu"), default="nc_p")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (derive_training_horizons(args.data_root) if args.dataset == "nc_p"
              else derive_xjtu_training_horizons(args.data_root))
    result["raw_data_root"] = str(args.data_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "training_device_count": result["training_device_count"],
        "median_training_device_max_rul": result["median_training_device_max_rul"],
        "horizons": result["horizons"],
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()
