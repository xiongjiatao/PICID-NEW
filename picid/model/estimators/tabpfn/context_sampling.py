"""Deterministic context sampling policies for TabPFN fit contexts."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


def unit_balanced_temporal_indices(
    unit_ids: np.ndarray | torch.Tensor,
    max_samples: int,
) -> np.ndarray:
    """Select a capped context with equal device weight and temporal coverage.

    Rows are grouped by their complete unit identifier. A near-equal quota is
    assigned to each unit, then row positions are selected at equal intervals
    within each unit's original row order. The returned global indices are
    sorted, preserving input order and feature/target alignment.
    """
    if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples <= 0:
        raise ValueError("max_samples must be a positive integer")

    if isinstance(unit_ids, torch.Tensor):
        unit_ids = unit_ids.detach().cpu().numpy()
    ids = np.asarray(unit_ids)
    if ids.ndim == 0:
        raise ValueError("unit_ids must have one row per fit sample")
    if ids.ndim == 1:
        ids = ids.reshape(-1, 1)
    else:
        ids = ids.reshape(ids.shape[0], -1)

    if ids.shape[0] == 0:
        raise ValueError("unit_ids cannot be empty")
    if ids.dtype.kind in "fc" and not np.isfinite(ids).all():
        raise ValueError("unit_ids cannot contain NaN or infinite values")

    rows_by_unit: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for row_idx, row in enumerate(ids):
        key = tuple(value.item() if isinstance(value, np.generic) else value for value in row)
        try:
            hash(key)
        except TypeError as exc:
            raise ValueError("unit_ids must contain scalar, hashable identifiers") from exc
        rows_by_unit[key].append(row_idx)

    n_rows = ids.shape[0]
    if n_rows <= max_samples:
        return np.arange(n_rows, dtype=np.int64)

    group_rows = list(rows_by_unit.values())
    if max_samples < len(group_rows):
        raise ValueError(
            f"max_samples={max_samples} cannot cover all {len(group_rows)} units"
        )
    allocations = [0] * len(group_rows)
    remaining = max_samples
    while remaining:
        active = [
            group_idx
            for group_idx, rows in enumerate(group_rows)
            if allocations[group_idx] < len(rows)
        ]
        if not active:
            raise RuntimeError("Could not allocate the requested fit context")
        base, remainder = divmod(remaining, len(active))
        if base == 0:
            for group_idx in active[:remainder]:
                allocations[group_idx] += 1
            remaining = 0
            continue
        for group_idx in active:
            addition = min(base, len(group_rows[group_idx]) - allocations[group_idx])
            allocations[group_idx] += addition
        remaining = max_samples - sum(allocations)

    selected: list[np.ndarray] = []
    for rows, quota in zip(group_rows, allocations, strict=True):
        if quota == len(rows):
            local = np.arange(len(rows), dtype=np.int64)
        elif quota == 1:
            local = np.array([(len(rows) - 1) // 2], dtype=np.int64)
        else:
            local = np.linspace(0, len(rows) - 1, num=quota, dtype=np.int64)
        selected.append(np.asarray(rows, dtype=np.int64)[local])

    return np.sort(np.concatenate(selected))
