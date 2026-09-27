"""Compare PICID's fixed N-CMAPSS scalers with NC-P train-unit statistics.

Only units 1--5 from each DS01/04/05/07 development partition contribute to
the train-only comparison. Validation (unit 6) and test (units 7--10) values are
read only as part of bounded HDF chunks and never enter accumulated moments.
This audit does not change the paper-fixed scaler used in the primary run.
"""

import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np

from picid.data.data_objects import NamedTransformInput
from picid.transforms.n_cmapss.n_cmapss_scalers import (
    N_CMAPSSDescriptorsScaler,
    N_CMAPSSFeaturesScaler,
)


class RunningStats:
    """Numerically stable population mean/std accumulator."""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            return
        batch_count = len(values)
        batch_mean = values.mean(axis=0)
        batch_m2 = np.square(values - batch_mean).sum(axis=0)
        if self.count == 0:
            self.count, self.mean, self.m2 = batch_count, batch_mean, batch_m2
            return
        delta = batch_mean - self.mean
        new_count = self.count + batch_count
        self.m2 += batch_m2 + np.square(delta) * self.count * batch_count / new_count
        self.mean += delta * batch_count / new_count
        self.count = new_count

    def result(self):
        if self.count == 0:
            raise ValueError("No training rows were selected for scaler audit")
        return {
            "count": self.count,
            "mean": self.mean.tolist(),
            "std": np.sqrt(self.m2 / self.count).tolist(),
        }


def _strings(dataset):
    values = np.asarray(dataset).reshape(-1)
    return [
        value.decode().strip() if isinstance(value, bytes) else str(value).strip()
        for value in values
    ]


def _fixed_stats(scaler_class, key, width):
    scaler = scaler_class(scaling="standard")
    zeros = np.zeros((1, width), dtype=np.float64)
    ones = np.ones((1, width), dtype=np.float64)
    z0 = scaler.transform_data(NamedTransformInput(**{key: zeros}), {})[0]
    z1 = scaler.transform_data(NamedTransformInput(**{key: ones}), {})[0]
    std = 1.0 / (z1 - z0)
    mean = -z0 * std
    return {"mean": mean.tolist(), "std": std.tolist()}


def audit(
    data_root: Path,
    sources=("01", "04", "05", "07"),
    train_units=range(1, 6),
    chunk_rows=500_000,
    file_hashes=None,
):
    sensor_stats = RunningStats()
    descriptor_stats = RunningStats()
    per_source = {}
    columns = {}
    for source in sources:
        path = data_root / "N-CMAPSS" / f"N-CMAPSS_DS{source}.h5"
        if not path.is_file():
            raise FileNotFoundError(path)
        sensor_source = RunningStats()
        descriptor_source = RunningStats()
        with h5py.File(path, "r") as hdf:
            required = ("X_s_dev", "W_dev", "A_dev", "X_s_var", "W_var", "A_var")
            absent = [key for key in required if key not in hdf]
            if absent:
                raise KeyError(f"{path} missing HDF datasets: {absent}")
            sensor_columns = _strings(hdf["X_s_var"])
            descriptor_columns = _strings(hdf["W_var"])
            aux_columns = _strings(hdf["A_var"])
            if "unit" not in aux_columns:
                raise KeyError(f"{path} A_var has no unit field: {aux_columns}")
            unit_column = aux_columns.index("unit")
            sensors, descriptors, auxiliary = hdf["X_s_dev"], hdf["W_dev"], hdf["A_dev"]
            if not (len(sensors) == len(descriptors) == len(auxiliary)):
                raise ValueError(f"Unaligned HDF arrays in {path}")
            if (
                len(sensor_columns) != sensors.shape[1]
                or len(descriptor_columns) != descriptors.shape[1]
            ):
                raise ValueError(f"Column metadata does not match HDF shapes in {path}")
            for start in range(0, len(auxiliary), chunk_rows):
                stop = min(start + chunk_rows, len(auxiliary))
                aux_chunk = auxiliary[start:stop]
                unit_ids = aux_chunk[:, unit_column]
                select = np.isin(unit_ids, list(train_units))
                if not select.any():
                    continue
                sensor_rows = sensors[start:stop][select]
                descriptor_rows = descriptors[start:stop][select]
                sensor_stats.update(sensor_rows)
                descriptor_stats.update(descriptor_rows)
                sensor_source.update(sensor_rows)
                descriptor_source.update(descriptor_rows)
        columns = {"features": sensor_columns, "descriptors": descriptor_columns}
        per_source[source] = {
            "file": str(path),
            "bytes": path.stat().st_size,
            "sha256": (file_hashes or {}).get(path.name),
            "selected_units": list(train_units),
            "features": sensor_source.result(),
            "descriptors": descriptor_source.result(),
        }
    fixed_features = _fixed_stats(
        N_CMAPSSFeaturesScaler, "features", len(columns["features"])
    )
    fixed_descriptors = _fixed_stats(
        N_CMAPSSDescriptorsScaler, "descriptors", len(columns["descriptors"])
    )

    def compare(fixed, fitted):
        std_ratio = np.asarray(fixed["std"]) / np.asarray(fitted["std"])
        mean_delta_in_train_std = (
            np.asarray(fixed["mean"]) - np.asarray(fitted["mean"])
        ) / np.asarray(fitted["std"])
        return {
            "fixed_mean": fixed["mean"],
            "fixed_std": fixed["std"],
            "train_only_mean": fitted["mean"],
            "train_only_std": fitted["std"],
            "fixed_std_over_train_std": std_ratio.tolist(),
            "fixed_mean_delta_in_train_std": mean_delta_in_train_std.tolist(),
        }

    feature_comparison = compare(fixed_features, sensor_stats.result())
    descriptor_comparison = compare(fixed_descriptors, descriptor_stats.result())
    feature_mean_delta = np.asarray(feature_comparison["fixed_mean_delta_in_train_std"])
    feature_std_ratio = np.asarray(feature_comparison["fixed_std_over_train_std"])
    descriptor_mean_delta = np.asarray(
        descriptor_comparison["fixed_mean_delta_in_train_std"]
    )
    descriptor_std_ratio = np.asarray(
        descriptor_comparison["fixed_std_over_train_std"]
    )
    all_mean_deltas = np.concatenate(
        [np.abs(feature_mean_delta), np.abs(descriptor_mean_delta)]
    )
    all_std_ratios = np.concatenate([feature_std_ratio, descriptor_std_ratio])

    return {
        "status": "FIXED_SCALERS_COMPARED_TO_NC_P_TRAIN_UNITS",
        "sources": list(sources),
        "train_units_per_source": list(train_units),
        "validation_units_excluded": [6],
        "test_units_excluded": [7, 8, 9, 10],
        "chunk_rows": chunk_rows,
        "columns": columns,
        "pooled_train_rows": {
            "features": sensor_stats.count,
            "descriptors": descriptor_stats.count,
        },
        "features": feature_comparison,
        "descriptors": descriptor_comparison,
        "comparison_summary": {
            "max_abs_mean_delta_train_std": float(all_mean_deltas.max()),
            "min_fixed_to_train_std_ratio": float(all_std_ratios.min()),
            "max_fixed_to_train_std_ratio": float(all_std_ratios.max()),
        },
        "per_source": per_source,
        "interpretation": "This checks numerical agreement only; it cannot recover the original population used to create benchmark-fixed constants.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extraction-status", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=500_000)
    args = parser.parse_args()
    started = time.time()
    extraction = json.loads(args.extraction_status.read_text())
    if extraction.get("status") != "DATA_READY_PROTOCOL_AUDIT_STILL_REQUIRED":
        raise RuntimeError("The extraction status is not DATA_READY")
    file_hashes = {item["local_name"]: item["sha256"] for item in extraction["files"]}
    report = audit(
        args.data_root, chunk_rows=args.chunk_rows, file_hashes=file_hashes
    )
    report["elapsed_seconds"] = time.time() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output),
                "pooled_train_rows": report["pooled_train_rows"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
