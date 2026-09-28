import numpy as np

from picid.evaluator.multiunit import MultiUnitEvaluator
from picid.pipeline.base import _collapse_single_task_unit_ids


def test_fit_predict_singleton_task_axis_is_removed_for_sample_ids():
    ids = np.array([[[1, 1], [1, 1], [4, 2], [4, 2]]])
    predictions = np.zeros((4, 1, 1))
    result = _collapse_single_task_unit_ids(ids, predictions)
    assert result.shape == (4, 2)
    np.testing.assert_array_equal(result, ids[0])
    ordinary = np.array([1, 1, 2, 2])
    assert _collapse_single_task_unit_ids(ordinary, predictions).shape == (4,)


def test_device_macro_metrics_are_the_mean_of_composite_ids():
    evaluator = MultiUnitEvaluator(metric_names=["mae"], task_type="regression")
    predictions = np.array([[[0.0]], [[2.0]], [[10.0]], [[14.0]]])
    targets = np.zeros((4, 1, 1))
    batched_ids = np.array([[[1, 1], [1, 1], [4, 2], [4, 2]]])
    evaluator.update({
        "predictions": predictions,
        "targets": targets,
        "unit_id": _collapse_single_task_unit_ids(batched_ids, predictions),
    })
    metrics = evaluator.compute(mode="test", epoch=0, step=0)
    assert metrics["mae_normalized_(1, 1)"] == 1.0
    assert metrics["mae_normalized_(4, 2)"] == 12.0
    assert metrics["mae_normalized_mean"] == 6.5
