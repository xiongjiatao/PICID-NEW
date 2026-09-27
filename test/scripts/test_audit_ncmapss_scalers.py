"""Guard the NC-P scaler comparison against validation/test-row contamination."""

import importlib.util
from pathlib import Path

import h5py
import numpy as np

spec = importlib.util.spec_from_file_location(
    "audit_ncmapss_scalers",
    Path(__file__).parents[2] / "scripts/research/audit_ncmapss_scalers.py",
)
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def test_scaler_audit_uses_only_training_unit_ids(tmp_path):
    directory = tmp_path / "N-CMAPSS"
    directory.mkdir()
    path = directory / "N-CMAPSS_DS01.h5"
    units = np.array([1, 2, 6, 7], dtype=np.int32)
    sensor_rows = np.array([1.0, 3.0, 100.0, 1000.0])
    descriptor_rows = np.array([2.0, 6.0, 200.0, 2000.0])
    with h5py.File(path, "w") as hdf:
        hdf.create_dataset("A_var", data=np.array([b"unit"]))
        hdf.create_dataset(
            "X_s_var", data=np.array([f"x{i}".encode() for i in range(14)])
        )
        hdf.create_dataset("W_var", data=np.array([f"w{i}".encode() for i in range(4)]))
        hdf.create_dataset("A_dev", data=units.reshape(-1, 1))
        hdf.create_dataset("X_s_dev", data=np.repeat(sensor_rows[:, None], 14, axis=1))
        hdf.create_dataset("W_dev", data=np.repeat(descriptor_rows[:, None], 4, axis=1))

    report = audit_module.audit(
        tmp_path, sources=("01",), train_units=(1, 2), chunk_rows=2
    )

    assert report["pooled_train_rows"] == {"features": 2, "descriptors": 2}
    np.testing.assert_allclose(report["features"]["train_only_mean"], np.full(14, 2.0))
    np.testing.assert_allclose(report["features"]["train_only_std"], np.full(14, 1.0))
    np.testing.assert_allclose(
        report["descriptors"]["train_only_mean"], np.full(4, 4.0)
    )
    np.testing.assert_allclose(report["descriptors"]["train_only_std"], np.full(4, 2.0))
    assert report["per_source"]["01"]["selected_units"] == [1, 2]
