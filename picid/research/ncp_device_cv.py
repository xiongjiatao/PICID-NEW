"""Grouped device-level resampling utilities for the NC-P development pool."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from picid.research.temporal_controls import stratified_timeline_indices


def nc_device_names(unit_ids: np.ndarray) -> np.ndarray:
    """Format source/unit identities without merging source-local unit numbers."""
    ids = np.asarray(unit_ids)
    if ids.ndim != 2 or ids.shape[1] != 2:
        raise ValueError("NC-P unit_ids must have shape (rows, 2): source and unit")
    if not np.isfinite(ids).all() or not np.equal(ids, np.floor(ids)).all():
        raise ValueError("NC-P source and unit IDs must be finite integers")
    return np.asarray(
        [f"DS{int(source):02d}-unit{int(unit):02d}" for source, unit in ids],
        dtype=object,
    )


def source_stratified_device_folds(
    unit_ids: np.ndarray,
    *,
    sources: tuple[int, ...] = (1, 4, 5, 7),
    local_units: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> list[dict[str, object]]:
    """Hold out one source-local unit per source in each deterministic fold."""
    names = nc_device_names(unit_ids)
    unique = sorted(set(names.tolist()))
    expected = {
        f"DS{source:02d}-unit{unit:02d}" for source in sources for unit in local_units
    }
    if set(unique) != expected:
        missing = sorted(expected - set(unique))
        extra = sorted(set(unique) - expected)
        raise ValueError(
            f"Unexpected NC-P device inventory; missing={missing}, extra={extra}"
        )

    folds: list[dict[str, object]] = []
    held_out_once: list[str] = []
    for fold_index, held_unit in enumerate(local_units):
        test_devices = [f"DS{source:02d}-unit{held_unit:02d}" for source in sources]
        train_devices = [name for name in unique if name not in test_devices]
        train_mask = np.isin(names, train_devices)
        test_mask = np.isin(names, test_devices)
        if np.any(train_mask & test_mask) or not np.all(train_mask | test_mask):
            raise ValueError("Fold assignment is not a complete, disjoint device split")
        folds.append(
            {
                "fold": fold_index,
                "held_out_local_unit": held_unit,
                "train_devices": train_devices,
                "test_devices": test_devices,
                "train_mask": train_mask,
                "test_mask": test_mask,
            }
        )
        held_out_once.extend(test_devices)
    if sorted(held_out_once) != unique:
        raise ValueError("Every NC-P development device must be held out exactly once")
    return folds


def balanced_strided_context_indices(
    units: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    stride: int,
    context_rows: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Sample a fixed context budget, water-filled across train devices.

    Devices with too few stride-eligible endpoints are capped at their available
    capacity. Their unused quota is redistributed as evenly as possible across
    the remaining devices. This keeps the total context budget fixed without
    duplicating endpoints or silently dropping short devices.
    """
    if stride < 1 or not units or context_rows < len(units):
        raise ValueError("Require positive stride and at least one context per device")
    names = sorted(units)
    candidates_by_device: dict[str, np.ndarray] = {}
    capacities: list[int] = []
    for name in names:
        features, targets = units[name]
        if not len(features) or len(features) != len(targets):
            raise ValueError(f"Require non-empty aligned features/targets for {name}")
        candidates = np.arange(0, len(features), stride, dtype=np.int64)
        if not len(candidates):
            raise ValueError(f"No stride-eligible context endpoint for {name}")
        candidates_by_device[name] = candidates
        capacities.append(len(candidates))

    if sum(capacities) < context_rows:
        raise ValueError(
            f"stride={stride} leaves only {sum(capacities)} eligible endpoints "
            f"across {len(names)} train devices; need {context_rows}"
        )

    # Max-min fair allocation subject to each device's eligible endpoint cap.
    lower, upper = 0, max(capacities)
    while lower < upper:
        level = (lower + upper + 1) // 2
        if sum(min(capacity, level) for capacity in capacities) <= context_rows:
            lower = level
        else:
            upper = level - 1
    allocation = [min(capacity, lower) for capacity in capacities]
    remaining = context_rows - sum(allocation)
    tie_order = np.random.default_rng(seed).permutation(len(names)).tolist()
    for rank in tie_order:
        if not remaining:
            break
        if allocation[rank] < capacities[rank]:
            allocation[rank] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("Water-filled context allocation did not use its budget")

    selected: dict[str, np.ndarray] = {}
    for rank, name in enumerate(names):
        candidates = candidates_by_device[name]
        count = allocation[rank]
        sampled = stratified_timeline_indices(len(candidates), count, seed + rank)
        selected[name] = candidates[sampled]
    return selected


def full_timeline_query_indices(length: int) -> np.ndarray:
    """Return every transformed-time query row in chronological order."""
    if length < 1:
        raise ValueError("length must be positive")
    return np.arange(length, dtype=np.int64)


def validate_full_timeline_query_rows(
    unit_ids: np.ndarray,
    query_times: np.ndarray,
    expected_counts: Mapping[str, int],
) -> None:
    """Require each listed device's full query timeline exactly once in order."""
    ids = np.asarray(unit_ids).astype(str).reshape(-1)
    times = np.asarray(query_times, dtype=np.int64).reshape(-1)
    if len(ids) != len(times) or not expected_counts:
        raise ValueError("query IDs, times, and expected device counts must be aligned")
    if set(np.unique(ids)) != set(expected_counts):
        raise ValueError("query device inventory differs from the registered split")
    for device, count in expected_counts.items():
        if count < 1:
            raise ValueError(f"Expected query count must be positive for {device}")
        observed = times[ids == device]
        if len(observed) != count or not np.array_equal(
            observed, np.arange(count, dtype=np.int64)
        ):
            raise ValueError(
                f"query timeline for {device} is incomplete, repeated, or reordered"
            )


def fold_critical_horizons(
    train_targets: np.ndarray,
    train_unit_ids: np.ndarray,
    *,
    native_units_per_target_unit: float,
    fractions: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> dict[str, float]:
    """Set warning/error horizons from training-device lifetimes in each fold."""
    targets = np.asarray(train_targets, dtype=np.float64).reshape(-1)
    names = nc_device_names(train_unit_ids)
    if len(targets) != len(names) or not len(targets):
        raise ValueError(
            "training targets and device IDs must be non-empty and aligned"
        )
    if not np.isfinite(targets).all() or native_units_per_target_unit <= 0:
        raise ValueError("targets must be finite and scale must be positive")
    device_lifetimes = [
        float(np.max(targets[names == device]) * native_units_per_target_unit)
        for device in sorted(set(names.tolist()))
    ]
    median_lifetime = float(np.median(device_lifetimes))
    if not np.isfinite(median_lifetime) or median_lifetime <= 0:
        raise ValueError("training-device median lifetime must be positive")
    return {
        f"{int(round(fraction * 100))}pct": median_lifetime * fraction
        for fraction in fractions
    }
