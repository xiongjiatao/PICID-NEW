import h5py
import numpy as np
import pytest

from scripts.research.derive_critical_horizons import read_training_device_max_rul
from scripts.research.evaluate_critical_phase import evaluate_arrays


def test_training_horizon_reads_only_registered_training_units(tmp_path):
    path = tmp_path / "N-CMAPSS_DS01.h5"
    units = np.repeat([1, 2, 6], 3)
    rul = np.concatenate(([100, 80, 60], [200, 180, 160], [1000, 900, 800]))
    with h5py.File(path, "w") as data:
        data.create_dataset("A_var", data=np.asarray([b"unit", b"cycle", b"Fc", b"hs"]))
        data.create_dataset("A_dev", data=np.column_stack((units, np.arange(9), units, units)))
        data.create_dataset("Y_dev", data=rul.reshape(-1, 1))

    result = read_training_device_max_rul(path, 1, train_units=(1, 2))
    assert result == {"DS01-unit01": 100.0, "DS01-unit02": 200.0}


def test_critical_scores_use_raw_rul_horizon_and_equal_device_macro():
    # Device 1 contributes two late-phase points; device 2 contributes one.
    ids = np.asarray([[1, 7], [1, 7], [4, 7]])
    target = np.asarray([5.0, 15.0, 5.0])
    pred = np.asarray([7.0, 17.0, 9.0])
    result = evaluate_arrays(pred, target, pred / 100, target / 100, ids, {"20pct": 10.0})

    assert result["points_per_device"] == {
        "DS01-unit07": {"20pct": 1},
        "DS04-unit07": {"20pct": 1},
    }
    assert result["metrics"]["critical_20pct_mae_denormalized_mean"] == pytest.approx(3.0)
    assert result["metrics"]["critical_20pct_rmse_denormalized_mean"] == pytest.approx(3.0)
