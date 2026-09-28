import awkward as ak
import numpy as np
from omegaconf import OmegaConf

from picid.data.data_objects import NamedTransformInput
from picid.research.n_cmapss import UnitIdentifierTransform, UnitWiseWindowedAggregationTransform
from picid.transforms.base_transforms.tabularizers import TimeseriesTabularizer
from picid.evaluator.multiunit import MultiUnitEvaluator
from picid.pipeline.base import _collapse_single_task_unit_ids


def test_window_aggregation_and_history_never_cross_engine_boundary():
    raw = ak.Array([
        [[0.0], [1.0], [2.0], [3.0], [4.0]],
        [[100.0], [101.0], [102.0], [103.0], [104.0], [105.0], [106.0]],
    ])
    raw = ak.to_regular(raw, axis=2)
    aggregation = UnitWiseWindowedAggregationTransform(
        window_size=2, step=2, aggregation="mean"
    )
    grouped, _ = aggregation.transform_multi_source(
        [NamedTransformInput(features=raw)],
        metadata={"apply_to_keys": ["features"]},
    )
    assert ak.to_list(grouped[0]["features"]) == [
        [[0.5], [2.5]], [[100.5], [102.5], [104.5]]
    ]

    tabularizer = TimeseriesTabularizer(
        select_features=OmegaConf.create([{"features": "history"}]),
        timestep_dimension=1,
        seq_len=2,
        label_len=0,
        pred_len=0,
        stride=1,
        padding_left_flag=False,
    )
    output, _ = tabularizer.transform_multi_source(
        [grouped[0]], metadata={"apply_to_keys": ["features"], "mode": "train"}
    )
    rows = np.asarray(output[0]).reshape(-1, 2)
    assert rows.shape == (3, 2)
    assert np.all(rows[0] < 10)
    assert np.all(rows[1:] > 90)


def test_composite_device_identity_survives_query_tabularization():
    units = ak.Array([[1, 1, 1], [2, 2, 2, 2]])
    sources = ak.Array([[1.0, 1.0, 1.0], [4.0, 4.0, 4.0, 4.0]])
    transform = UnitIdentifierTransform()
    grouped, _ = transform.transform_multi_source(
        [NamedTransformInput(unit=units, n_DS=sources)],
        metadata={"apply_to_keys": ["unit", "n_DS"]},
    )
    assert ak.to_list(grouped[0]["unit_id"]) == [
        [[1.0, 1.0]] * 3,
        [[4.0, 2.0]] * 4,
    ]

    tabularizer = TimeseriesTabularizer(
        select_features=OmegaConf.create([{"unit_id": "present"}]),
        timestep_dimension=1,
        seq_len=1,
        label_len=0,
        pred_len=0,
        stride=1,
        padding_left_flag=True,
    )
    ids, _ = tabularizer.transform_multi_source(
        [grouped[0]], metadata={"apply_to_keys": ["unit_id"], "mode": "test"}
    )
    np.testing.assert_array_equal(ids[0].reshape(-1, 2),
                                  [[1, 1]] * 3 + [[4, 2]] * 4)


def test_multiunit_evaluator_reports_device_macro_mean():
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
