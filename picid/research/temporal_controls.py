"""Device-isolated window representations for controlled PHM comparisons."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def contiguous_unit_slices(unit_ids: np.ndarray) -> list[tuple[str, slice]]:
    """Return contiguous row ranges for each composite unit identifier.

    Unit IDs may be one-dimensional or a row-aligned matrix (for example
    ``[datasource_id, source_local_unit]``). A unit that reappears after another
    unit is rejected because silently joining it would create invalid histories.
    """
    ids = np.asarray(unit_ids)
    if ids.ndim == 1:
        rows = [(str(value),) for value in ids]
    elif ids.ndim == 2:
        rows = [tuple(str(value) for value in row) for row in ids]
    else:
        raise ValueError("unit_ids must have shape (N,) or (N, K)")
    if not rows:
        raise ValueError("unit_ids must be non-empty")

    ranges: list[tuple[str, slice]] = []
    seen: set[tuple[str, ...]] = set()
    start = 0
    current = rows[0]
    for index, key in enumerate(rows[1:], start=1):
        if key != current:
            if current in seen:
                raise ValueError(f"Unit ID reappears in non-contiguous rows: {current}")
            seen.add(current)
            ranges.append(("-".join(current), slice(start, index)))
            start, current = index, key
    if current in seen:
        raise ValueError(f"Unit ID reappears in non-contiguous rows: {current}")
    ranges.append(("-".join(current), slice(start, len(rows))))
    return ranges


def stratified_timeline_indices(length: int, count: int, seed: int) -> np.ndarray:
    """Select one deterministic random point from each equal timeline stratum."""
    if length < 1 or count < 1 or count > length:
        raise ValueError("Require 1 <= count <= length")
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, length, count + 1, dtype=np.int64)
    selected = np.asarray(
        [rng.integers(edges[i], edges[i + 1]) for i in range(count)],
        dtype=np.int64,
    )
    return np.sort(selected)


def history_windows(
    features: np.ndarray, end_indices: np.ndarray, window: int
) -> np.ndarray:
    """Build causal histories ending at each query, edge-padding at unit start."""
    values = np.asarray(features, dtype=np.float32)
    ends = np.asarray(end_indices, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or not len(values) or window < 1:
        raise ValueError(
            "features must be non-empty (time, feature) and window positive"
        )
    if not len(ends) or np.any(ends < 0) or np.any(ends >= len(values)):
        raise ValueError("end_indices must be non-empty valid timeline indices")
    offsets = np.arange(window, dtype=np.int64) - (window - 1)
    row_indices = np.clip(ends[:, None] + offsets[None, :], 0, len(values) - 1)
    return values[row_indices]


def multiscale_summary(histories: np.ndarray) -> np.ndarray:
    """Summarize nested suffix scales with mean, standard deviation, and slope."""
    values = np.asarray(histories, dtype=np.float32)
    if values.ndim != 3 or not values.shape[0] or not values.shape[1]:
        raise ValueError("histories must have non-empty (sample, time, feature) shape")
    scales = sorted(
        {1, 5, 10, 25, values.shape[1]}.intersection(range(1, values.shape[1] + 1))
    )
    summaries: list[np.ndarray] = []
    for width in scales:
        suffix = values[:, -width:, :]
        summaries.append(suffix.mean(axis=1))
        summaries.append(suffix.std(axis=1))
        if width == 1:
            slope = np.zeros_like(summaries[-1])
        else:
            time = np.arange(width, dtype=np.float32)
            centered = time - time.mean()
            denom = np.sum(centered**2)
            slope = np.einsum("t,ntf->nf", centered, suffix) / denom
        summaries.append(slope.astype(np.float32, copy=False))
    return np.concatenate(summaries, axis=1).astype(np.float32, copy=False)


def device_balanced_sample(
    units: Mapping[str, tuple[np.ndarray, np.ndarray]],
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Sample equal, timeline-stratified rows from every device.

    ``units`` maps a stable device name to ``(features, target)``. The final
    smaller units are rejected rather than silently receiving less weight.
    """
    if not units or count < len(units):
        raise ValueError("count must allocate at least one row per device")
    names = sorted(units)
    base, remainder = divmod(count, len(names))
    sampled_x: list[np.ndarray] = []
    sampled_y: list[np.ndarray] = []
    sampled_ids: list[np.ndarray] = []
    for rank, name in enumerate(names):
        x, y = units[name]
        x = np.asarray(x)
        y = np.asarray(y).reshape(-1)
        if len(x) != len(y):
            raise ValueError(f"feature/target length mismatch for {name}")
        n = base + (rank < remainder)
        indices = stratified_timeline_indices(len(x), n, seed + rank)
        sampled_x.append(x[indices])
        sampled_y.append(y[indices])
        sampled_ids.append(np.full(n, name, dtype=object))
    return (
        np.concatenate(sampled_x),
        np.concatenate(sampled_y),
        np.concatenate(sampled_ids),
        names,
    )


def regression_metrics_by_device(
    predictions: np.ndarray,
    targets: np.ndarray,
    unit_ids: np.ndarray,
    horizons: Mapping[str, float],
    *,
    critical_values: np.ndarray | None = None,
) -> dict[str, object]:
    """Compute equal-device regression errors with explicit critical-phase units."""
    pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    ids = np.asarray(unit_ids).reshape(-1)
    phase_values = (
        target
        if critical_values is None
        else np.asarray(critical_values, dtype=np.float64).reshape(-1)
    )
    if not (len(pred) == len(target) == len(ids)) or not len(pred):
        raise ValueError(
            "prediction, target, and unit arrays must be non-empty and aligned"
        )
    if len(phase_values) != len(pred):
        raise ValueError("critical_values must align with predictions and targets")
    if (
        not np.isfinite(pred).all()
        or not np.isfinite(target).all()
        or not np.isfinite(phase_values).all()
    ):
        raise ValueError("prediction and target values must be finite")
    output: dict[str, object] = {"device_count": int(np.unique(ids).size)}
    per_device: dict[str, dict[str, float | int]] = {}
    for unit in np.unique(ids):
        mask = ids == unit
        error = pred[mask] - target[mask]
        per_device[str(unit)] = {
            "query_count": int(mask.sum()),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
        }
        for horizon_name, horizon in horizons.items():
            phase = mask & (phase_values <= horizon)
            if phase.any():
                phase_error = pred[phase] - target[phase]
                per_device[str(unit)][f"{horizon_name}_mae"] = float(
                    np.mean(np.abs(phase_error))
                )
                per_device[str(unit)][f"{horizon_name}_query_count"] = int(phase.sum())
            else:
                per_device[str(unit)][f"{horizon_name}_mae"] = float("nan")
                per_device[str(unit)][f"{horizon_name}_query_count"] = 0
    output["per_device"] = per_device
    metric_names = ["mae", "rmse", *[f"{name}_mae" for name in horizons]]
    for metric in metric_names:
        values = [
            row[metric] for row in per_device.values() if np.isfinite(row[metric])
        ]
        output[f"device_macro_{metric}"] = (
            float(np.mean(values)) if values else float("nan")
        )
    return output
