"""Aggregate fixed-configuration seed results and device-level uncertainty."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED_SEEDS = (72, 88, 101)


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = location - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _source_group(dataset: str, device: str) -> str:
    if dataset == "nc_p":
        match = re.fullmatch(r"(DS\d+)-unit\d+", device)
        if not match:
            raise ValueError(f"Invalid NC-P device identifier: {device}")
        return match.group(1)
    if dataset == "xjtu" and re.fullmatch(r"bearing[^/]+", device):
        return "all_test_bearings"
    raise ValueError(f"Unsupported device identifier for {dataset}: {device}")


def aggregate_seed_metrics(
    reports: list[dict[str, Any]],
    bootstrap_replicates: int = 10000,
    bootstrap_seed: int = 20260928,
) -> dict[str, Any]:
    """Validate three seed reports and separate seed and device uncertainty."""
    if bootstrap_replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required")
    if len(reports) != len(EXPECTED_SEEDS):
        raise ValueError(f"Expected exactly {len(EXPECTED_SEEDS)} seed reports")

    by_seed = {int(report["seed"]): report for report in reports}
    if tuple(sorted(by_seed)) != EXPECTED_SEEDS:
        raise ValueError(f"Expected seeds {EXPECTED_SEEDS}, found {tuple(sorted(by_seed))}")
    first = by_seed[EXPECTED_SEEDS[0]]
    dataset = first["dataset"]
    config_hash = first["source_protocol_sha256"]
    if any(report.get("dataset") != dataset for report in reports):
        raise ValueError("Seed reports have different datasets")
    if any(report.get("source_protocol_sha256") != config_hash for report in reports):
        raise ValueError("Seed reports do not share one frozen configuration digest")
    if any(report.get("aggregation") != "equal_device_macro" for report in reports):
        raise ValueError("All reports must use equal-device macro aggregation")

    devices = set(first["per_device"])
    metric_names = set(first["metrics"])
    device_metric_names = {
        metric[:-5] if metric.endswith("_mean") else metric for metric in metric_names
    }
    for report in reports:
        if set(report["per_device"]) != devices:
            raise ValueError("Seed reports contain different device sets")
        if set(report["metrics"]) != metric_names:
            raise ValueError("Seed reports contain different macro metrics")
        for device_values in report["per_device"].values():
            if set(device_values) != device_metric_names:
                raise ValueError("Per-device metric sets differ from macro metric bases")
            if any(not math.isfinite(float(value)) for value in device_values.values()):
                raise ValueError("Non-finite per-device metric")
        if any(not math.isfinite(float(value)) for value in report["metrics"].values()):
            raise ValueError("Non-finite macro metric")

    ordered_devices = sorted(devices)
    source_groups: dict[str, list[str]] = defaultdict(list)
    for device in ordered_devices:
        source_groups[_source_group(dataset, device)].append(device)

    seed_summary: dict[str, Any] = {}
    device_summary: dict[str, Any] = {}
    source_summary: dict[str, Any] = {}
    device_means: dict[str, dict[str, float]] = {}
    for metric in sorted(metric_names):
        values = [float(by_seed[seed]["metrics"][metric]) for seed in EXPECTED_SEEDS]
        seed_summary[metric] = {
            "mean": sum(values) / len(values),
            "sample_std": _sample_std(values),
            "values_by_seed": {str(seed): value for seed, value in zip(EXPECTED_SEEDS, values)},
        }

        device_means[metric] = {}
        device_summary[metric] = {}
        device_metric = metric[:-5] if metric.endswith("_mean") else metric
        for device in ordered_devices:
            per_seed = [float(by_seed[seed]["per_device"][device][device_metric])
                        for seed in EXPECTED_SEEDS]
            mean = sum(per_seed) / len(per_seed)
            device_means[metric][device] = mean
            device_summary[metric][device] = {
                "mean_across_seeds": mean,
                "sample_std_across_seeds": _sample_std(per_seed),
            }

        source_summary[metric] = {}
        for group, group_devices in sorted(source_groups.items()):
            source_values = [device_means[metric][device] for device in group_devices]
            source_summary[metric][group] = {
                "mean_across_devices_and_seeds": sum(source_values) / len(source_values),
                "device_count": len(group_devices),
            }

    rng = random.Random(bootstrap_seed)
    ci: dict[str, Any] = {}
    for metric in sorted(metric_names):
        draws = []
        for _ in range(bootstrap_replicates):
            sampled = []
            for group_devices in source_groups.values():
                sampled.extend(
                    device_means[metric][rng.choice(group_devices)]
                    for _ in range(len(group_devices))
                )
            draws.append(sum(sampled) / len(sampled))
        ci[metric] = {
            "estimate": sum(device_means[metric].values()) / len(ordered_devices),
            "lower_95_percentile": _quantile(draws, 0.025),
            "upper_95_percentile": _quantile(draws, 0.975),
            "resampling": "within_data_source_with_equal_device_weights" if dataset == "nc_p"
            else "test_bearings_with_equal_device_weights",
            "device_count": len(ordered_devices),
            "source_groups": len(source_groups),
            "replicates": bootstrap_replicates,
        }

    return {
        "protocol": "three_seed_fixed_configuration",
        "dataset": dataset,
        "seeds": list(EXPECTED_SEEDS),
        "source_protocol_sha256": config_hash,
        "aggregation": "equal_device_macro",
        "device_count": len(ordered_devices),
        "seed_uncertainty": "sample standard deviation across the three fixed seeds (ddof=1)",
        "device_uncertainty": (
            "95% percentile bootstrap over seed-averaged device scores; resample engines "
            "within each NC-P source, then weight sources equally"
            if dataset == "nc_p"
            else "95% percentile bootstrap over seed-averaged test-bearing scores"
        ),
        "seed_summary": seed_summary,
        "device_bootstrap_95_ci": ci,
        "per_source": source_summary,
        "per_device_seed_summary": device_summary,
        "input_report_sha256": {
            str(seed): hashlib.sha256(
                json.dumps(by_seed[seed], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            for seed in EXPECTED_SEEDS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260928)
    args = parser.parse_args()

    reports = [json.loads(path.read_text()) for path in args.input]
    result = aggregate_seed_metrics(reports, args.bootstrap_replicates, args.bootstrap_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "dataset": result["dataset"],
        "seeds": result["seeds"],
        "device_count": result["device_count"],
        "metrics": result["seed_summary"],
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()
