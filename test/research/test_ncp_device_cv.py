"""Protocol checks for NC-P source-stratified device-level validation."""

import numpy as np
import pytest

from picid.research.ncp_device_cv import (
    balanced_strided_context_indices,
    common_query_indices,
    fold_critical_horizons,
    source_stratified_device_folds,
)
from picid.research.temporal_controls import history_windows


def _device_ids() -> np.ndarray:
    return np.asarray(
        [
            (source, unit)
            for source in (1, 4, 5, 7)
            for unit in (1, 2, 3, 4, 5)
            for _ in range(3)
        ],
        dtype=np.int64,
    )


def test_source_stratified_folds_hold_out_each_engine_once() -> None:
    unit_ids = _device_ids()
    folds = source_stratified_device_folds(unit_ids)
    held_out: list[str] = []

    for fold_index, fold in enumerate(folds):
        train_mask = np.asarray(fold["train_mask"])
        test_mask = np.asarray(fold["test_mask"])
        assert fold["fold"] == fold_index
        assert train_mask.sum() == 16 * 3
        assert test_mask.sum() == 4 * 3
        assert not np.any(train_mask & test_mask)
        assert np.all(train_mask | test_mask)
        held_out.extend(fold["test_devices"])

    assert len(held_out) == 20
    assert len(set(held_out)) == 20


def test_fold_horizons_use_only_training_devices() -> None:
    unit_ids = _device_ids()
    targets = np.asarray([float(unit) for _, unit in unit_ids], dtype=np.float64)
    targets[unit_ids[:, 1] == 1] = 1_000.0
    fold = source_stratified_device_folds(unit_ids)[0]
    train_mask = np.asarray(fold["train_mask"])
    horizons = fold_critical_horizons(
        targets[train_mask],
        unit_ids[train_mask],
        native_units_per_target_unit=100.0,
    )
    assert horizons == {"5pct": 17.5, "10pct": 35.0, "20pct": 70.0}

    wrong_shape = np.asarray(unit_ids[:, 1])
    with pytest.raises(ValueError, match=r"shape \(rows, 2\)"):
        fold_critical_horizons(targets, wrong_shape, native_units_per_target_unit=100.0)


def test_context_endpoints_respect_stride_and_balance_devices() -> None:
    units = {
        "DS01-unit01": (np.zeros((600, 2), dtype=np.float32), np.zeros(600)),
        "DS04-unit01": (np.ones((600, 2), dtype=np.float32), np.zeros(600)),
    }
    selected = balanced_strided_context_indices(
        units, stride=5, context_rows=20, seed=72
    )
    assert {len(values) for values in selected.values()} == {10}
    assert all(np.all(values % 5 == 0) for values in selected.values())
    assert len(np.unique(selected["DS01-unit01"])) == 10

    history_a = history_windows(units["DS01-unit01"][0], np.asarray([0, 1]), 5)
    history_b = history_windows(units["DS04-unit01"][0], np.asarray([0, 1]), 5)
    assert np.all(history_a == 0)
    assert np.all(history_b == 1)


def test_common_query_grid_is_deterministic_and_spans_timeline() -> None:
    expected = np.asarray([0, 2, 4, 6, 9])
    np.testing.assert_array_equal(common_query_indices(10, 5), expected)
    np.testing.assert_array_equal(common_query_indices(10, 20), np.arange(10))


def test_context_sampler_rejects_stride_with_too_few_candidates() -> None:
    units = {"DS01-unit01": (np.zeros((20, 2)), np.zeros(20))}
    with pytest.raises(ValueError, match="need 5"):
        balanced_strided_context_indices(units, stride=10, context_rows=5, seed=72)
