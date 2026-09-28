"""Compute paper-style NASA RUL scores from saved raw-unit predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr


def nasa_scores(predictions, targets, unit_ids, expected_device_count=16):
    pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    ids = np.asarray(unit_ids)
    if ids.ndim != 2 or ids.shape[1] != 2 or len(pred) != len(target) or len(ids) != len(pred):
        raise ValueError("NASA score inputs require aligned rows and composite source/unit IDs")
    difference = pred - target
    scores = np.empty_like(difference)
    underprediction = difference < 0
    scores[underprediction] = np.exp(-difference[underprediction] / 13) - 1
    scores[~underprediction] = np.exp(difference[~underprediction] / 10) - 1
    per_device = {}
    for source, unit in np.unique(ids.astype(np.int64), axis=0):
        device = f"DS{source:02d}-unit{unit:02d}"
        mask = (ids == (source, unit)).all(axis=1)
        per_device[device] = float(np.mean(scores[mask]))
    if expected_device_count is not None and len(per_device) != expected_device_count:
        raise ValueError(
            f"Expected {expected_device_count} NC-P test engines, found {len(per_device)}"
        )
    return {
        "metrics": {
            "nasa_score_denormalized_mean": float(np.mean(list(per_device.values())))
        },
        "per_device": {
            device: {"nasa_score_denormalized": value}
            for device, value in per_device.items()
        },
        "row_weighted_nasa_score_denormalized": float(np.mean(scores)),
        "row_count": len(scores),
        "device_count": len(per_device),
    }


def _digest(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(72, 88, 101), required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with xr.open_dataset(args.predictions) as dataset:
        values = nasa_scores(dataset["preds"].values, dataset["targets"].values,
                             dataset["unit_ids"].values)
    result = {
        "dataset": "nc_p",
        "seed": args.seed,
        "source_protocol_sha256": args.config_sha256,
        "aggregation": "equal_device_macro",
        "formula": "mean(exp(-error/13)-1) if error<0 else mean(exp(error/10)-1); error=prediction-target",
        "prediction_sha256": _digest(args.predictions),
        **values,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"seed": args.seed, "metrics": result["metrics"],
                      "row_weighted": result["row_weighted_nasa_score_denormalized"],
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
