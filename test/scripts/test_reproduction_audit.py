"""Regression checks for research protocol inspection (no training/data needed)."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "research_audit",
    Path(__file__).parents[2] / "scripts/research/audit_reproduction.py",
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_ignored_last_aggregation_is_blocked():
    cfg = {
        "transforms": {
            "target": {
                "transform": {
                    "_target_": "x.WindowedAggregationTransform",
                    "aggregation": "last",
                }
            }
        }
    }
    assert (
        audit.inspect_config(cfg, {"aggregation_alias_supported": False})[0]["code"]
        == "IGNORED_AGGREGATION"
    )


def test_explicit_last_aggregation_has_no_mismatch():
    cfg = {
        "transforms": {
            "target": {
                "transform": {
                    "_target_": "x.WindowedAggregationTransform",
                    "agg": "last",
                    "aggregation": "last",
                }
            }
        }
    }
    assert audit.inspect_config(cfg, {"aggregation_alias_supported": False}) == []


def test_snapshot_detects_change_and_ignores_environment(tmp_path):
    source = tmp_path / "a.txt"
    source.write_text("baseline")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "b").write_text("ignored")
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "archive.zip.part").write_bytes(b"in-progress download")
    before = audit.snapshot(tmp_path)
    source.write_text("changed")
    assert before != audit.snapshot(tmp_path)
    assert list(before) == ["a.txt"]


def test_released_aggregation_runtime_counterexample():
    import numpy as np
    from picid.data.data_objects import NamedTransformInput
    from picid.transforms.base_transforms.subsample import WindowedAggregationTransform

    values = np.array([[1.0], [4.0], [10.0]])
    released = WindowedAggregationTransform(window_size=3, step=3, aggregation="last")
    corrected = WindowedAggregationTransform(window_size=3, step=3, agg="last")
    actual = released.transform_data(NamedTransformInput(target=values.copy()), {})[
        "target"
    ]
    desired = corrected.transform_data(NamedTransformInput(target=values.copy()), {})[
        "target"
    ]
    np.testing.assert_allclose(actual, [[5.0]])
    np.testing.assert_allclose(desired, [[10.0]])
