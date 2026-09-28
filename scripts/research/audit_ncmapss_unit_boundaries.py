"""Audit that transformed NC-P rows remain aligned with source/device IDs."""
import argparse
import json
from pathlib import Path
import sys

import awkward as ak
import joblib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from picid.research.chunks import atomic_json  # noqa: E402

EXPECTED = {"train": set(range(1, 6)), "val": {6}, "test": {7, 8, 9, 10}}
SOURCE_TO_DS = {"train": [1, 4, 5, 7], "val": [1, 4, 5, 7], "test": [1, 4, 5, 7]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True,
                        help="Preprocessed MultiSource_concepts_N-CMAPSS/data.pkl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    container = joblib.load(args.cache)
    report = {"protocol": "NC-P DS01/04/05/07; unit labels are source-local",
              "units": [], "status": "passed"}
    for split, expected_units in EXPECTED.items():
        features = container["features"][split]
        targets = container["rul"][split]
        units = container["unit"][split]
        datasets = container["n_DS"][split]
        if not (len(features) == len(targets) == len(units) == len(datasets) == 4):
            raise AssertionError(f"Expected four source segments in {split}")
        for index, ds in enumerate(SOURCE_TO_DS[split]):
            unit_groups = ak.to_list(units[index])
            ds_groups = ak.to_list(datasets[index])
            row_count = int(features[index].shape[1])
            target_count = int(targets[index].shape[1])
            if row_count != target_count or row_count != sum(map(len, unit_groups)):
                raise AssertionError(f"Feature/RUL/device row alignment failed for DS{ds:02d} {split}")
            if len(unit_groups) != len(ds_groups):
                raise AssertionError("Engine identity groups misaligned")
            for unit_values, ds_values in zip(unit_groups, ds_groups):
                unit_ids = sorted(set(map(int, unit_values)))
                ds_ids = sorted(set(map(int, ds_values)))
                if len(unit_ids) != 1 or unit_ids[0] not in expected_units:
                    raise AssertionError(f"Unexpected engine group DS{ds:02d}/{unit_ids}")
                if ds_ids != [ds]:
                    raise AssertionError(f"Datasource ID merged across sources: {ds_ids}")
                report["units"].append({"split": split, "device_id": f"DS{ds:02d}-unit{unit_ids[0]:02d}",
                                        "source_local_unit_id": unit_ids[0],
                                        "rows": len(unit_values),
                                        "aligned_feature_rows": row_count,
                                        "aligned_target_rows": target_count})
    report["device_count_by_split"] = {
        split: sum(row["split"] == split for row in report["units"])
        for split in EXPECTED
    }
    atomic_json(args.output, report)
    print(json.dumps({"status": report["status"], "devices": len(report["units"]),
                      "device_count_by_split": report["device_count_by_split"]}))


if __name__ == "__main__":
    main()
