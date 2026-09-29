"""Validated readers for saved PICID prediction artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def load_normalized_predictions(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load normalized predictions, targets, and composite unit IDs from NetCDF."""
    required = {"preds_normalized", "targets_normalized", "unit_ids"}
    with xr.open_dataset(path) as dataset:
        missing = required.difference(dataset.data_vars)
        if missing:
            raise ValueError(
                f"Prediction dataset is missing variables: {sorted(missing)}"
            )
        unit_ids = np.asarray(dataset["unit_ids"].values)
        predictions = np.asarray(dataset["preds_normalized"].values)
        targets = np.asarray(dataset["targets_normalized"].values)

    if unit_ids.ndim != 2 or unit_ids.shape[1] != 2:
        raise ValueError(
            f"Expected two-column composite unit IDs, got {unit_ids.shape}"
        )
    predictions = predictions.reshape(len(unit_ids), -1)
    targets = targets.reshape(len(unit_ids), -1)
    if predictions.shape[1] != 1 or targets.shape[1] != 1:
        raise ValueError("Expected one scalar normalized prediction and target per row")
    if not np.isfinite(unit_ids).all() or not np.isfinite(predictions).all():
        raise ValueError("Prediction data and unit IDs must be finite")
    if not np.isfinite(targets).all():
        raise ValueError("Normalized targets must be finite")
    if not np.equal(unit_ids, np.floor(unit_ids)).all():
        raise ValueError("Composite unit IDs must be integer-valued")
    return (
        unit_ids.astype(np.int64, copy=False),
        targets.reshape(-1),
        predictions.reshape(-1),
    )
