import numpy as np
import pytest

from scripts.research.summarize_nasa_score import nasa_scores


def test_nasa_score_is_computed_per_device_and_macro_averaged():
    predictions = np.asarray([9.0, 11.0, 12.0])
    targets = np.asarray([10.0, 10.0, 10.0])
    ids = np.asarray([[1, 7], [1, 7], [4, 7]])

    result = nasa_scores(predictions, targets, ids, expected_device_count=2)

    under = np.exp(1 / 13) - 1
    over_one = np.exp(1 / 10) - 1
    over_two = np.exp(2 / 10) - 1
    assert result["per_device"]["DS01-unit07"]["nasa_score_denormalized"] == pytest.approx(
        (under + over_one) / 2
    )
    assert result["per_device"]["DS04-unit07"]["nasa_score_denormalized"] == pytest.approx(
        over_two
    )
    assert result["metrics"]["nasa_score_denormalized_mean"] == pytest.approx(
        ((under + over_one) / 2 + over_two) / 2
    )
    assert result["row_weighted_nasa_score_denormalized"] == pytest.approx(
        (under + over_one + over_two) / 3
    )
