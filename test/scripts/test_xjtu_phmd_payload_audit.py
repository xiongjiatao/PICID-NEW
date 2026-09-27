"""Synthetic checks for XJTU PHMD RUL and health-index semantics."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "audit_xjtu_payload",
    Path(__file__).parents[2] / "scripts/research/audit_xjtu_phmd_payload.py",
)
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def test_phmd_reverse_acquisition_index_matches_life_table():
    summary = audit_module.derive_phmd_acquisition_rul([32769] * 5)
    assert summary["matches_reverse_acquisition_index"]
    assert summary["each_acquisition_has_constant_rul"]
    assert summary["rul_per_acquisition"] == [4, 3, 2, 1, 0]
    assert summary["total_data_rows"] == 5 * 32768


def test_partial_acquisition_does_not_claim_one_label_per_file():
    summary = audit_module.derive_phmd_acquisition_rul([32769, 10])
    assert not summary["matches_reverse_acquisition_index"]
    assert not summary["each_acquisition_has_constant_rul"]
