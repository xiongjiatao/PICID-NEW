"""Load XJTU-SY through the locked PHMD/PICID path and audit labels/splits.

The source directory is read-only and is expected at
``<data-root>/phmd_cache/datasets/XJTU-SY/train``. No download is attempted.
The report checks all 15 bearings, acquisition counts, RUL direction and the
paper's PHMD train/validation/test assignment without fitting a model.
"""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import re
import time

from picid.data.datasources.phmd_xjtu_sy import (
    SAMPLE_RANGE_SIZE,
    XJTU_SY_SPLIT_ASSIGNMENTS_BY_MODE,
)
from picid.transforms.bearings.health_index import DEFAULT_TOTAL_LIFE_LOOKUP

EXPECTED_SPLITS = {
    "train": ["1_3", "1_4", "2_1", "2_4", "2_5", "3_1", "3_2", "3_3"],
    "val": ["1_1", "1_2", "3_5"],
    "test": ["1_5", "2_2", "2_3", "3_4"],
}


def count_csv_rows(path, chunk_bytes=8 * 1024 * 1024):
    """Count physical CSV rows with bounded memory, without parsing signals."""
    newline_count = 0
    last_byte = b""
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            newline_count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if not last_byte:
        return 0
    return newline_count + int(last_byte != b"\n")


def numeric_csv_files(bearing_dir):
    """Return acquisition files ordered exactly as the PHMD reader orders them."""
    files = list(bearing_dir.glob("*.csv"))
    if any(not re.fullmatch(r"\d+", path.stem) for path in files):
        raise ValueError(f"Unexpected non-numeric acquisition filename in {bearing_dir}")
    return sorted(files, key=lambda path: int(path.stem))


def derive_phmd_acquisition_rul(physical_line_counts):
    """Reproduce PHMD labels at acquisition boundaries without materializing rows.

    PHMD's ``pandas.read_csv`` consumes one physical line as the CSV header;
    it then labels the remaining rows with ``floor(index / 32768)`` reversed.
    """
    data_rows = [line_count - 1 for line_count in physical_line_counts]
    if any(row_count <= 0 for row_count in data_rows):
        raise ValueError("Each acquisition CSV must contain a header and data rows")
    total_rows = sum(data_rows)
    offset = 0
    labels = []
    boundaries_constant = True
    for row_count in data_rows:
        first = (total_rows - 1 - offset) // SAMPLE_RANGE_SIZE
        last = (total_rows - 1 - (offset + row_count - 1)) // SAMPLE_RANGE_SIZE
        boundaries_constant &= first == last
        labels.append(int(first))
        offset += row_count
    expected = list(range(len(data_rows) - 1, -1, -1))
    return {
        "data_rows_per_acquisition": data_rows,
        "total_data_rows": total_rows,
        "rul_per_acquisition": labels,
        "each_acquisition_has_constant_rul": bool(boundaries_constant),
        "matches_reverse_acquisition_index": (
            boundaries_constant
            and all(rows == SAMPLE_RANGE_SIZE for rows in data_rows)
            and labels == expected
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    cache_dir = data_root / "phmd_cache"
    raw_path = cache_dir / "datasets" / "XJTU-SY" / "train"
    if not raw_path.is_dir():
        raise FileNotFoundError(
            f"Expected existing XJTU source at {raw_path}; refusing to download data."
        )

    started = time.time()
    all_units = sorted(
        path.name for path in raw_path.iterdir() if path.is_dir()
    )
    configured_split = XJTU_SY_SPLIT_ASSIGNMENTS_BY_MODE["phmd_split"]
    if configured_split != EXPECTED_SPLITS:
        raise AssertionError(
            "PICID phmd_split differs from the published PHMD split table: "
            f"{configured_split!r} != {EXPECTED_SPLITS!r}"
        )
    report = {
        "status": "XJTU_PAYLOAD_AUDIT",
        "phmd_version": version("phmd"),
        "source_path": str(raw_path.resolve()),
        "source_is_symlink": raw_path.is_symlink(),
        "audit_mode": "streaming_csv_row_count_plus_locked_phmd_label_formula",
        "phmd_rul_formula": f"floor(global_row_index / {SAMPLE_RANGE_SIZE}), reversed over the bearing timeline",
        "phmd_reader_source": "phmd/readers/xjtu-sy.py: read_files()",
        "expected_split": EXPECTED_SPLITS,
        "observed": {},
        "all_checks_pass": True,
    }
    life_lookup = DEFAULT_TOTAL_LIFE_LOOKUP["XJTU-SY"]
    expected_all_units = sorted(
        f"Bearing{unit}" for split in EXPECTED_SPLITS.values() for unit in split
    )
    report["all_checks_pass"] &= all_units == expected_all_units
    report["observed_all_units"] = all_units
    report["expected_all_units"] = expected_all_units
    for split, expected_units in EXPECTED_SPLITS.items():
        entries = []
        for suffix in expected_units:
            unit_name = f"Bearing{suffix}"
            bearing_dir = raw_path / unit_name
            if not bearing_dir.is_dir():
                raise FileNotFoundError(f"Missing bearing directory: {bearing_dir}")
            files = numeric_csv_files(bearing_dir)
            physical_line_counts = [count_csv_rows(path) for path in files]
            minutes = len(files)
            phmd_labels = derive_phmd_acquisition_rul(physical_line_counts)
            all_files_are_one_minute = all(
                count == SAMPLE_RANGE_SIZE + 1 for count in physical_line_counts
            )
            unit_id = tuple(int(value) for value in suffix.split("_"))
            total_life = life_lookup.get(unit_id)
            if total_life is None:
                raise KeyError(f"Missing total-life entry for {suffix}")
            rul_trace = phmd_labels["rul_per_acquisition"]
            first_rul = float(rul_trace[0])
            last_rul = float(rul_trace[-1])
            checks = {
                "unit_name": unit_name,
                "unit_id": list(unit_id),
                "acquisition_files": minutes,
                "table_total_life_minutes": int(total_life),
                "physical_line_count_min": min(physical_line_counts, default=0),
                "physical_line_count_max": max(physical_line_counts, default=0),
                **phmd_labels,
                "all_acquisitions_have_32768_rows": all_files_are_one_minute,
                "rul_first_acquisition": first_rul,
                "rul_last_acquisition": last_rul,
                "matches_phmd_reverse_index": phmd_labels[
                    "matches_reverse_acquisition_index"
                ],
                "acquisition_count_matches_life_table": minutes == total_life,
                "first_hi": first_rul / total_life,
                "last_hi": last_rul / total_life,
            }
            entries.append(checks)
            print(
                f"audited {split}/{unit_name}: files={minutes} "
                f"physical_lines={checks['physical_line_count_min']}.."
                f"{checks['physical_line_count_max']} "
                f"RUL={first_rul}..{last_rul}",
                flush=True,
            )
            report["all_checks_pass"] &= (
                checks["all_acquisitions_have_32768_rows"]
                and checks["matches_phmd_reverse_index"]
                and checks["acquisition_count_matches_life_table"]
            )
        report["observed"][split] = {"units": entries}
    report["elapsed_seconds"] = time.time() - started
    report["status"] = "PASS" if report["all_checks_pass"] else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output),
                "splits": {
                    k: [unit["unit_name"] for unit in v["units"]]
                    for k, v in report["observed"].items()
                },
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0 if report["all_checks_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
