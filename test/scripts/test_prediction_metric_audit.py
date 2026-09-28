import numpy as np
import xarray as xr

from scripts.research.audit_prediction_metrics import audit_arrays


def test_prediction_export_reconciles_per_device_metrics():
    predictions = np.asarray([[[0.2]], [[0.4]], [[0.5]]], dtype=np.float32)
    targets = np.asarray([[[0.1]], [[0.5]], [[0.2]]], dtype=np.float32)
    ids = np.asarray([[1, 7], [1, 7], [4, 7]], dtype=np.float32)
    raw = predictions.reshape(-1) * 100
    labels = targets.reshape(-1) * 100
    report = {
        "dataset": "nc_p",
        "seed": 72,
        "per_device": {
            "DS01-unit07": {
                "mae_denormalized": 10.0,
                "rmse_denormalized": 10.0,
                "mse_denormalized": 100.0,
                "mae_normalized": 0.1,
                "rmse_normalized": 0.1,
                "mse_normalized": 0.01,
            },
            "DS04-unit07": {
                "mae_denormalized": 30.0,
                "rmse_denormalized": 30.0,
                "mse_denormalized": 900.0,
                "mae_normalized": 0.3,
                "rmse_normalized": 0.3,
                "mse_normalized": 0.09,
            },
        },
    }
    dataset = xr.Dataset({
        "preds": (("sample", "time", "feature"), raw.reshape(-1, 1, 1)),
        "targets": (("sample", "time", "feature"), labels.reshape(-1, 1, 1)),
        "preds_normalized": (("sample", "time", "feature"), predictions),
        "targets_normalized": (("sample", "time", "feature"), targets),
        "unit_ids": (("sample", "unit_dim_1"), ids),
    })

    result = audit_arrays(dataset, report)

    assert result["row_count"] == 3
    assert result["device_count"] == 2
    assert max(result["max_abs_metric_difference"].values()) < 1e-6
