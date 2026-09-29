"""Protocol checks for NC-P source-stratified device-level validation."""

import numpy as np
import pytest

from picid.research.ncp_device_cv import (
    balanced_strided_context_indices,
    full_timeline_query_indices,
    fold_critical_horizons,
    source_stratified_device_folds,
    validate_full_timeline_query_rows,
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


def test_query_indices_include_every_timeline_row_in_order() -> None:
    np.testing.assert_array_equal(full_timeline_query_indices(10), np.arange(10))
    np.testing.assert_array_equal(full_timeline_query_indices(1), np.asarray([0]))
    validate_full_timeline_query_rows(
        np.asarray(["DS01-unit01"] * 3 + ["DS04-unit01"] * 2),
        np.asarray([0, 1, 2, 0, 1]),
        {"DS01-unit01": 3, "DS04-unit01": 2},
    )


@pytest.mark.parametrize(
    ("unit_ids", "query_times"),
    [
        (["DS01-unit01", "DS01-unit01", "DS04-unit01"], [0, 2, 0]),
        (["DS01-unit01", "DS01-unit01", "DS04-unit01"], [1, 0, 0]),
        (["DS01-unit01", "DS01-unit01", "DS04-unit02"], [0, 1, 0]),
    ],
)
def test_full_query_audit_rejects_missing_reordered_or_wrong_devices(
    unit_ids: list[str], query_times: list[int]
) -> None:
    with pytest.raises(ValueError):
        validate_full_timeline_query_rows(
            np.asarray(unit_ids),
            np.asarray(query_times),
            {"DS01-unit01": 2, "DS04-unit01": 1},
        )


def test_context_sampler_redistributes_short_device_quota_without_duplicates() -> None:
    units = {
        "DS01-unit01": (np.zeros((21, 2)), np.zeros(21)),
        "DS04-unit01": (np.ones((201, 2)), np.zeros(201)),
        "DS05-unit01": (np.full((201, 2), 2), np.zeros(201)),
    }
    selected = balanced_strided_context_indices(
        units, stride=10, context_rows=20, seed=72
    )
    counts = {name: len(indices) for name, indices in selected.items()}
    assert counts["DS01-unit01"] == 3
    assert sorted(counts.values()) == [3, 8, 9]
    assert sum(counts.values()) == 20
    assert all(np.all(indices % 10 == 0) for indices in selected.values())
    assert all(len(np.unique(indices)) == len(indices) for indices in selected.values())
    replayed = balanced_strided_context_indices(
        units, stride=10, context_rows=20, seed=72
    )
    for name in selected:
        np.testing.assert_array_equal(selected[name], replayed[name])


def test_context_sampler_fails_if_stride_caps_cannot_fill_budget() -> None:
    units = {"DS01-unit01": (np.zeros((20, 2)), np.zeros(20))}
    with pytest.raises(ValueError, match="only 2 eligible endpoints.*need 5"):
        balanced_strided_context_indices(units, stride=10, context_rows=5, seed=72)


def test_paper_context_budget_is_fully_allocated_under_short_device_caps() -> None:
    units = {
        f"DS{source:02d}-unit{unit:02d}": (
            np.zeros((length, 1), dtype=np.float32),
            np.zeros(length, dtype=np.float32),
        )
        for source in (1, 4, 5, 7)
        for unit, length in enumerate((5_950, 6_450, 7_050, 8_100), start=1)
    }
    selected = balanced_strided_context_indices(
        units, stride=50, context_rows=2048, seed=72
    )
    counts = np.asarray([len(indices) for indices in selected.values()])
    assert counts.sum() == 2048
    assert counts.min() >= 119
    assert counts.max() <= 132
    assert np.all([np.all(indices % 50 == 0) for indices in selected.values()])
