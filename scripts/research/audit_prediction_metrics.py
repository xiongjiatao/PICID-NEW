"""Reconcile saved prediction arrays with the per-device metric report."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr


CHECKED_METRICS = {
    "mae_denormalized": lambda error: float(np.mean(np.abs(error))),
    "rmse_denormalized": lambda error: float(np.sqrt(np.mean(error ** 2))),
    "mse_denormalized": lambda error: float(np.mean(error ** 2)),
    "mae_normalized": lambda error: float(np.mean(np.abs(error))),
    "rmse_normalized": lambda error: float(np.sqrt(np.mean(error ** 2))),
    "mse_normalized": lambda error: float(np.mean(error ** 2)),
}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_arrays(dataset, report: dict, atol: float = 1e-4, rtol: float = 1e-6):
    if not np.isfinite([atol, rtol]).all() or min(atol, rtol) < 0:
        raise ValueError("Audit tolerances must be finite and nonnegative")
    pred = dataset["preds"].values.reshape(-1)
    target = dataset["targets"].values.reshape(-1)
    pred_norm = dataset["preds_normalized"].values.reshape(-1)
    target_norm = dataset["targets_normalized"].values.reshape(-1)
    ids = dataset["unit_ids"].values
    if ids.ndim != 2 or ids.shape[1] != 2:
        raise ValueError(f"Expected composite source/unit IDs, received {ids.shape}")
    for name, values in (("predictions", pred), ("targets", target),
                         ("normalized predictions", pred_norm),
                         ("normalized targets", target_norm), ("device IDs", ids)):
        if not values.size or not np.isfinite(values).all():
            raise ValueError(f"Empty or non-finite {name}")
    if not np.equal(ids, np.floor(ids)).all():
        raise ValueError("Device IDs must be integers")
    ids = ids.astype(np.int64)
    if not (len(pred) == len(target) == len(pred_norm) == len(target_norm) == len(ids)):
        raise ValueError("Saved predictions, targets and IDs have different row counts")
    device_rows = {}
    max_difference = {metric: 0.0 for metric in CHECKED_METRICS}
    for source, unit in np.unique(ids, axis=0):
        device = (f"bearing{source}_{unit}" if report["dataset"] == "xjtu"
                  else f"DS{source:02d}-unit{unit:02d}")
        if device not in report["per_device"]:
            raise ValueError(f"Saved predictions contain unexpected device {device}")
        mask = (ids == (source, unit)).all(axis=1)
        device_rows[device] = int(mask.sum())
        raw_error = np.asarray(pred[mask] - target[mask], dtype=np.float64)
        norm_error = np.asarray(pred_norm[mask] - target_norm[mask], dtype=np.float64)
        observed = {
            key: compute(raw_error)
            for key, compute in CHECKED_METRICS.items()
            if "denormalized" in key
        }
        observed.update({
            key: compute(norm_error)
            for key, compute in CHECKED_METRICS.items()
            if "normalized" in key and "denormalized" not in key
        })
        for metric, value in observed.items():
            reference = float(report["per_device"][device][metric])
            if not np.isfinite([value, reference]).all():
                raise ValueError(f"Non-finite metric: {device}/{metric}")
            difference = abs(value - reference)
            max_difference[metric] = max(max_difference[metric], difference)
            tolerance = atol + rtol * abs(float(report["per_device"][device][metric]))
            if difference > tolerance:
                raise ValueError(
                    f"{device}/{metric} differs from report by {difference:.8g} > {tolerance:.8g}"
                )
    if set(device_rows) != set(report["per_device"]):
        raise ValueError("Prediction device set differs from the per-device report")
    return {
        "row_count": len(pred),
        "device_count": len(device_rows),
        "device_row_count": device_rows,
        "checked_metrics": sorted(CHECKED_METRICS),
        "max_abs_metric_difference": max_difference,
        "atol": atol,
        "rtol": rtol,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args()
    report = json.loads(args.metrics.read_text())
    with xr.open_dataset(args.predictions) as dataset:
        audit = audit_arrays(dataset, report, args.atol, args.rtol)
    audit.update({
        "dataset": report["dataset"],
        "seed": report["seed"],
        "prediction_sha256": _digest(args.predictions),
        "metric_report_sha256": _digest(args.metrics),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
