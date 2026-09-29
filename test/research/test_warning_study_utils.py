import numpy as np
import pytest
import xarray as xr
import json

from picid.research.temporal_controls import (
    contiguous_unit_slices,
    device_balanced_sample,
    history_windows,
    multiscale_summary,
    regression_metrics_by_device,
    stratified_timeline_indices,
)
from picid.research.warning_metrics import (
    alarm_metrics,
    apply_platt_calibrator,
    fit_platt_calibrator,
    probability_metrics,
    select_empirical_fpr_threshold,
    warning_labels_from_rul,
)
from picid.research.warning_probabilities import (
    residual_ecdf_score,
    tabdpt_distribution_cdf,
    tabdpt_distribution_mean,
)
from picid.research.prediction_io import load_normalized_predictions


def test_prediction_io_reads_scalar_variables_and_composite_ids_from_netcdf(tmp_path):
    path = tmp_path / "predictions.nc"
    xr.Dataset(
        {
            "preds_normalized": (("sample", "time", "feature"), [[[0.2]], [[0.8]]]),
            "targets_normalized": (("sample", "time", "feature"), [[[0.1]], [[0.9]]]),
            "unit_ids": (("sample", "unit_dim"), [[1, 5], [3, 4]]),
        }
    ).to_netcdf(path, engine="scipy")

    ids, targets, predictions = load_normalized_predictions(path)

    np.testing.assert_array_equal(ids, [[1, 5], [3, 4]])
    np.testing.assert_allclose(targets, [0.1, 0.9])
    np.testing.assert_allclose(predictions, [0.2, 0.8])


def test_undefined_manifest_metrics_are_written_as_json_null(tmp_path):
    from scripts.research.run_xjtu_warning_study import _atomic_json

    path = tmp_path / "manifest.json"
    _atomic_json(path, {"defined": 0.5, "undefined": float("nan")})

    assert json.loads(path.read_text()) == {"defined": 0.5, "undefined": None}


def test_composite_unit_boundaries_are_preserved_and_reappearance_rejected():
    ids = np.array([[1, 1], [1, 1], [1, 2], [1, 2], [2, 1]])
    assert contiguous_unit_slices(ids) == [
        ("1-1", slice(0, 2)),
        ("1-2", slice(2, 4)),
        ("2-1", slice(4, 5)),
    ]
    with pytest.raises(ValueError, match="reappears"):
        contiguous_unit_slices(np.array([[1, 1], [1, 2], [1, 1]]))


def test_histories_are_causal_and_left_edge_padded_within_device():
    x = np.arange(10, dtype=np.float32).reshape(5, 2)
    histories = history_windows(x, np.array([0, 2, 4]), window=3)
    np.testing.assert_array_equal(histories[0], [x[0], x[0], x[0]])
    np.testing.assert_array_equal(histories[1], [x[0], x[1], x[2]])
    np.testing.assert_array_equal(histories[2], [x[2], x[3], x[4]])
    with pytest.raises(ValueError, match="valid timeline"):
        history_windows(x, np.array([5]), window=2)


def test_stratified_device_sample_is_deterministic_and_balanced():
    units = {
        "long": (np.arange(100).reshape(-1, 1), np.arange(100)),
        "short": (np.arange(30).reshape(-1, 1), np.arange(30)),
    }
    x1, y1, ids1, _ = device_balanced_sample(units, 12, seed=9)
    x2, y2, ids2, _ = device_balanced_sample(units, 12, seed=9)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(ids1, ids2)
    assert np.unique(ids1, return_counts=True)[1].tolist() == [6, 6]
    np.testing.assert_array_equal(
        stratified_timeline_indices(30, 6, seed=9),
        stratified_timeline_indices(30, 6, seed=9),
    )


def test_multiscale_statistics_have_fixed_width_and_finite_slopes():
    x = np.arange(20, dtype=np.float32).reshape(1, 10, 2)
    result = multiscale_summary(x)
    assert result.shape == (1, 3 * 3 * 2)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[0, 10:12], [2, 2])


def test_warning_labels_use_native_rul_and_macro_errors_weight_devices_equally():
    labels = warning_labels_from_rul(np.array([10, 2, 0]), {"early": 5})
    np.testing.assert_array_equal(labels["early"], [False, True, True])
    metrics = regression_metrics_by_device(
        np.array([0, 0, 0, 10]),
        np.array([0, 0, 0, 0]),
        np.array(["short", "short", "short", "long"]),
        {"critical": 0.5},
    )
    assert metrics["device_macro_mae"] == 5.0
    assert metrics["device_macro_critical_mae"] == 5.0


def test_critical_phase_uses_explicit_native_units_not_normalized_regression_targets():
    metrics = regression_metrics_by_device(
        predictions=np.array([0.0, 0.4, 0.1]),
        targets=np.array([0.0, 0.5, 0.0]),
        unit_ids=np.array(["bearing-a"] * 3),
        horizons={"5pct": 5.0},
        critical_values=np.array([10.0, 4.0, 1.0]),
    )
    row = metrics["per_device"]["bearing-a"]
    assert row["5pct_query_count"] == 2
    assert row["5pct_mae"] == pytest.approx(0.1)


def test_calibration_threshold_uses_device_macro_fpr_and_probabilities_are_finite():
    scores = np.array([0.1] * 100 + [0.2, 0.8, 0.9, 0.95])
    labels = np.zeros(len(scores), dtype=int)
    labels[-2:] = 1
    ids = np.array(["long"] * 100 + ["short"] * 4)
    threshold, fpr = select_empirical_fpr_threshold(scores, labels, ids, target_fpr=0.1)
    assert threshold >= 0.8
    assert fpr <= 0.1
    model = fit_platt_calibrator(scores, labels, ids, score_is_probability=True)
    probabilities = apply_platt_calibrator(model, scores, score_is_probability=True)
    metrics = probability_metrics(probabilities, labels, ids)
    assert 0 <= metrics["device_macro_brier"] <= 1
    assert metrics["device_count"] == 2


def test_alarm_episodes_use_test_rul_only_for_events_and_lead_time():
    rul = np.array([10, 8, 5, 4, 3, 0], dtype=float)
    labels = warning_labels_from_rul(rul, {"h": 5})["h"]
    result = alarm_metrics(
        scores=np.array([0.8, 0.8, 0.1, 0.9, 0.9, 0.1]),
        labels=labels,
        rul_at_query=rul,
        unit_ids=np.array(["bearing-a"] * 6),
        query_times=np.arange(6),
        threshold=0.5,
    )
    row = result.per_device["bearing-a"]
    assert row["false_alarm_episode_count"] == 1
    assert row["detected_in_warning_window"] is True
    assert row["lead_time_intervals"] == 4.0
    assert result.metrics["mean_alarm_episodes_per_device"] == 2.0


def test_alarm_starting_before_warning_is_not_retroactively_a_detection():
    rul = np.array([7, 6, 5, 4, 3, 2], dtype=float)
    labels = warning_labels_from_rul(rul, {"h": 5})["h"]
    result = alarm_metrics(
        scores=np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1]),
        labels=labels,
        rul_at_query=rul,
        unit_ids=np.array(["bearing-b"] * 6),
        query_times=np.arange(6),
        threshold=0.5,
    )
    row = result.per_device["bearing-b"]
    assert row["false_alarm_episode_count"] == 1
    assert row["detected_in_warning_window"] is False


def test_tabdpt_piecewise_uniform_distribution_mean_cdf_and_residual_ecdf():
    logits = np.log(np.array([[0.25, 0.75], [0.5, 0.5]]))
    borders = np.array([0.0, 1.0, 3.0])
    np.testing.assert_allclose(tabdpt_distribution_mean(logits, borders), [1.625, 1.25])
    np.testing.assert_allclose(
        tabdpt_distribution_cdf(logits, borders, 1.0), [0.25, 0.5]
    )
    residuals = {"a": np.array([-0.1, 0.1]), "b": np.array([0.2])}
    score = residual_ecdf_score(residuals, np.array([0.5]), 0.61)
    np.testing.assert_allclose(score, [0.5])
    score_without_a = residual_ecdf_score(
        residuals, np.array([0.5]), 0.61, exclude_device="a"
    )
    np.testing.assert_allclose(score_without_a, [0.0])
