import pytest

from scripts.research.aggregate_seed_metrics import aggregate_seed_metrics


def _report(seed, shift=0.0):
    devices = {}
    for source in (1, 4, 5, 7):
        for unit in (7, 8, 9, 10):
            devices[f"DS{source:02d}-unit{unit:02d}"] = {"mae_normalized": 0.1 + shift}
    return {
        "dataset": "nc_p",
        "seed": seed,
        "source_protocol_sha256": "frozen-config",
        "aggregation": "equal_device_macro",
        "metrics": {"mae_normalized_mean": 0.1 + shift},
        "per_device": devices,
    }


def test_separates_seed_std_from_source_stratified_device_interval():
    result = aggregate_seed_metrics([_report(72), _report(88, 0.01), _report(101, 0.02)], 500)

    metric = result["seed_summary"]["mae_normalized_mean"]
    assert metric["mean"] == pytest.approx(0.11)
    assert metric["sample_std"] == pytest.approx(0.01)
    assert result["device_count"] == 16
    assert result["device_bootstrap_95_ci"]["mae_normalized_mean"]["source_groups"] == 4
    assert result["per_source"]["mae_normalized_mean"]["DS01"]["device_count"] == 4


def test_rejects_seed_reports_with_different_device_sets():
    third = _report(101)
    third["per_device"].pop("DS07-unit10")

    with pytest.raises(ValueError, match="different device sets"):
        aggregate_seed_metrics([_report(72), _report(88), third], 500)
