"""Score selected-model RUL errors inside training-defined critical horizons."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scores(pred, target):
    error = np.asarray(pred, dtype=np.float64).reshape(-1) - np.asarray(target, dtype=np.float64).reshape(-1)
    return float(np.mean(np.abs(error))), float(np.sqrt(np.mean(error ** 2)))


def evaluate_arrays(prediction, target, prediction_normalized, target_normalized,
                    unit_ids, horizons):
    """Return per-device and equal-device macro scores for each fixed horizon."""
    pred = np.asarray(prediction).reshape(-1)
    target = np.asarray(target).reshape(-1)
    pred_norm = np.asarray(prediction_normalized).reshape(-1)
    target_norm = np.asarray(target_normalized).reshape(-1)
    ids = np.asarray(unit_ids)
    if ids.ndim != 2 or ids.shape[1] != 2:
        raise ValueError(f"Expected composite (source, unit) identifiers; got {ids.shape}")
    ids = ids.astype(np.int64)
    if not (len(pred) == len(target) == len(pred_norm) == len(target_norm) == len(ids)):
        raise ValueError("Prediction, label, and device ID lengths differ")
    if not all(np.isfinite(values).all() for values in (pred, target, pred_norm, target_norm)):
        raise ValueError("Prediction or label array contains non-finite values")

    per_device = {}
    point_counts = {}
    for source, unit in np.unique(ids, axis=0):
        device = f"DS{source:02d}-unit{unit:02d}"
        device_mask = (ids == (source, unit)).all(axis=1)
        scores = {}
        counts = {}
        for horizon_name, threshold in horizons.items():
            mask = device_mask & (target <= float(threshold))
            count = int(mask.sum())
            if count == 0:
                raise ValueError(f"No {horizon_name} queries for {device}; review frozen horizon")
            counts[horizon_name] = count
            for scale, values_pred, values_target in (
                ("denormalized", pred, target),
                ("normalized", pred_norm, target_norm),
            ):
                mae, rmse = _scores(values_pred[mask], values_target[mask])
                prefix = f"critical_{horizon_name}"
                scores[f"{prefix}_mae_{scale}"] = mae
                scores[f"{prefix}_rmse_{scale}"] = rmse
        per_device[device] = scores
        point_counts[device] = counts

    metric_names = sorted(next(iter(per_device.values())))
    macro = {
        f"{metric}_mean": float(np.mean([per_device[device][metric] for device in per_device]))
        for metric in metric_names
    }
    return {"metrics": macro, "per_device": per_device, "points_per_device": point_counts}


def evaluate_prediction_file(prediction_path: Path, horizon_manifest: dict,
                             seed: int, config_sha256: str):
    with xr.open_dataset(prediction_path) as dataset:
        result = evaluate_arrays(
            dataset["preds"].values,
            dataset["targets"].values,
            dataset["preds_normalized"].values,
            dataset["targets_normalized"].values,
            dataset["unit_ids"].values,
            {name: value["rul_threshold"]
             for name, value in horizon_manifest["horizons"].items()},
        )
    if len(result["per_device"]) != 16:
        raise ValueError(f"Expected 16 NC-P test engines, found {len(result['per_device'])}")
    return {
        "dataset": "nc_p",
        "seed": seed,
        "aggregation": "equal_device_macro",
        "source_protocol_sha256": config_sha256,
        "horizon_manifest_sha256": hashlib.sha256(
            json.dumps(horizon_manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "horizons": horizon_manifest["horizons"],
        "raw_rul_unit": horizon_manifest["raw_rul_unit"],
        "test_device_count": len(result["per_device"]),
        "test_prediction_sha256": _sha256(prediction_path),
        **result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--horizons", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(72, 88, 101), required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_prediction_file(
        args.predictions,
        json.loads(args.horizons.read_text()),
        args.seed,
        args.config_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"seed": args.seed, "metrics": result["metrics"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
