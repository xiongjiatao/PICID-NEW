import numpy as np
import pytest
import torch

from picid.model.estimators.tabpfn.context_sampling import (
    unit_balanced_temporal_indices,
)


def test_unit_balanced_temporal_sampling_covers_each_unit_and_time_span():
    unit_ids = torch.tensor([[0]] * 6 + [[1]] * 6)

    indices = unit_balanced_temporal_indices(unit_ids, max_samples=4)

    np.testing.assert_array_equal(indices, [0, 5, 6, 11])


def test_unit_balanced_temporal_sampling_redistributes_small_unit_quota():
    unit_ids = np.array([0] + [1] * 9 + [2] * 9)

    indices = unit_balanced_temporal_indices(unit_ids, max_samples=8)
    sampled_ids = unit_ids[indices]

    assert len(indices) == 8
    assert np.array_equal(np.unique(sampled_ids, return_counts=True)[1], [1, 4, 3])
    assert np.all(indices[:-1] < indices[1:])


def test_unit_balanced_temporal_sampling_is_deterministic_and_noop_below_cap():
    unit_ids = np.array([[0, 1]] * 2 + [[0, 2]] * 3)

    first = unit_balanced_temporal_indices(unit_ids, max_samples=20)
    second = unit_balanced_temporal_indices(unit_ids, max_samples=20)

    np.testing.assert_array_equal(first, np.arange(5))
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("limit", [0, -1, True])
def test_unit_balanced_temporal_sampling_rejects_invalid_limit(limit):
    with pytest.raises(ValueError, match="positive integer"):
        unit_balanced_temporal_indices(np.array([0, 1]), max_samples=limit)


def test_unit_balanced_temporal_sampling_rejects_missing_unit_ids():
    with pytest.raises(ValueError, match="NaN or infinite"):
        unit_balanced_temporal_indices(np.array([0.0, np.nan]), max_samples=1)


def test_unit_balanced_temporal_sampling_rejects_budget_smaller_than_unit_count():
    with pytest.raises(ValueError, match="cannot cover all 3 units"):
        unit_balanced_temporal_indices(np.array([0, 1, 2]), max_samples=2)
