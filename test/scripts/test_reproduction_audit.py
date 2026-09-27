"""Regression checks for research protocol inspection (no training/data needed)."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "research_audit",
    Path(__file__).parents[2] / "scripts/research/audit_reproduction.py",
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_historical_aggregation_bug_is_detected():
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


def test_current_aggregation_contract_accepts_both_names():
    contract = audit.aggregation_contract(audit.ROOT)
    assert "agg" in contract["parameters"]
    assert "aggregation" in contract["parameters"]
    assert contract["aggregation_alias_supported"] is True


def test_paper_intent_aggregation_is_resolved_explicitly():
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
    resolved = audit.resolve_aggregation_fields(cfg, "paper_intent_fixed")
    assert resolved == [
        {
            "transform": "target",
            "requested": "last",
            "effective": "last",
            "protocol": "paper_intent_fixed",
        }
    ]
    assert cfg["transforms"]["target"]["transform"]["agg"] == "last"
    assert "aggregation" not in cfg["transforms"]["target"]["transform"]


def test_legacy_protocol_records_historical_mean_default():
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
    resolution = audit.resolve_aggregation_fields(cfg, "released_behavior_legacy")
    assert resolution[0]["requested"] == "last"
    assert resolution[0]["effective"] == "mean"


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


def test_last_aggregation_alias_matches_canonical_parameter():
    import numpy as np
    from picid.data.data_objects import NamedTransformInput
    from picid.transforms.base_transforms.subsample import WindowedAggregationTransform

    values = np.array([[1.0], [4.0], [10.0]])
    alias = WindowedAggregationTransform(window_size=3, step=3, aggregation="last")
    corrected = WindowedAggregationTransform(window_size=3, step=3, agg="last")
    actual = alias.transform_data(NamedTransformInput(target=values.copy()), {})[
        "target"
    ]
    desired = corrected.transform_data(NamedTransformInput(target=values.copy()), {})[
        "target"
    ]
    np.testing.assert_allclose(actual, [[10.0]])
    np.testing.assert_allclose(desired, [[10.0]])


def test_source_dataset_modules_are_hashed(tmp_path):
    source = tmp_path / "picid" / "data" / "datasets"
    source.mkdir(parents=True)
    (source / "base.py").write_text("source")
    assert "picid/data/datasets/base.py" in audit.snapshot(tmp_path)
