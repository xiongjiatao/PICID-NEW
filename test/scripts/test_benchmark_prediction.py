import numpy as np
import pytest

from scripts.research.benchmark_prediction import (
    _predict_tabdpt,
    _tabdpt_context_size,
    prepare_benchmark_inputs,
)


def test_tabdpt120_benchmark_preserves_native_context_and_batch_api():
    calls = []

    class Model:
        def predict(self, queries, **kwargs):
            calls.append(kwargs)
            return np.zeros(len(queries))

    context_size = _tabdpt_context_size("tabdpt120")
    assert context_size is None
    _predict_tabdpt(Model(), np.zeros((5, 2)), 32, True, context_size)
    assert calls == [{"n_ensembles": 8, "seed": 72, "batch_size": 32}]


def test_older_tabdpt_benchmark_retains_explicit_context_cap():
    assert _tabdpt_context_size("tabdpt") == 2048
    assert _tabdpt_context_size("tabdpt130") == 2048


def test_prepare_benchmark_inputs_uses_row_aligned_unit_balanced_indices():
    train_features = np.arange(24).reshape(12, 2)
    train_targets = train_features[:, 0] * 10
    train_unit_ids = np.array([[0, 1]] * 6 + [[0, 2]] * 6)
    val_features = np.arange(36).reshape(12, 3)
    val_unit_ids = np.array([[1, 1]] * 6 + [[1, 2]] * 6)

    X, y, Q, selection = prepare_benchmark_inputs(
        train_features,
        train_targets,
        val_features,
        train_unit_ids=train_unit_ids,
        val_unit_ids=val_unit_ids,
        query_rows=4,
        max_fit_samples=4,
    )

    expected = np.array([0, 5, 6, 11])
    np.testing.assert_array_equal(X, train_features[expected])
    np.testing.assert_array_equal(y, train_targets[expected])
    np.testing.assert_array_equal(Q, val_features[expected])
    assert selection["fit_selection"] == "unit_balanced_temporal"
    assert selection["query_selection"] == "unit_balanced_temporal"
    assert selection["fit_unit_count"] == 2
    assert selection["query_unit_count"] == 2


def test_prepare_benchmark_inputs_uses_fixed_prefix_without_metadata():
    train_features = np.arange(20).reshape(10, 2)
    train_targets = np.arange(10)
    val_features = np.arange(30).reshape(10, 3)

    X, y, Q, selection = prepare_benchmark_inputs(
        train_features,
        train_targets,
        val_features,
        train_rows=4,
        query_rows=3,
    )

    np.testing.assert_array_equal(X, train_features[:4])
    np.testing.assert_array_equal(y, train_targets[:4])
    np.testing.assert_array_equal(Q, val_features[:3])
    assert selection["fit_selection"] == "chronological_prefix"
    assert selection["query_selection"] == "chronological_prefix"


def test_prepare_benchmark_inputs_rejects_misaligned_targets():
    with pytest.raises(ValueError, match="different row counts"):
        prepare_benchmark_inputs(
            np.ones((4, 2)),
            np.ones(3),
            np.ones((4, 2)),
        )


def test_prepare_benchmark_inputs_requires_unit_ids_for_context_cap():
    with pytest.raises(ValueError, match="unit_id metadata is required"):
        prepare_benchmark_inputs(
            np.ones((4, 2)),
            np.ones(4),
            np.ones((4, 2)),
            max_fit_samples=2,
        )


def test_prepare_benchmark_inputs_rejects_misaligned_unit_ids():
    with pytest.raises(ValueError, match="not row-aligned"):
        prepare_benchmark_inputs(
            np.ones((4, 2)),
            np.ones(4),
            np.ones((4, 2)),
            train_unit_ids=np.array([0, 0, 1]),
            max_fit_samples=2,
        )
