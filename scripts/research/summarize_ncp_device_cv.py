#!/usr/bin/env python3
"""Aggregate pre-registered NC-P grouped-CV manifests and device predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SEEDS = (72, 88, 101, 666, 226688)
METRICS = ("mae", "rmse", "5pct_mae", "10pct_mae", "20pct_mae")
BASELINE = "w1_s1"


def _bootstrap_device_difference(
    candidate: dict[str, float], baseline: dict[str, float], *, replicates: int = 10_000
) -> dict[str, float]:
    devices = sorted(candidate)
    by_source: dict[str, list[str]] = {}
    for device in devices:
        source = device.split("-", maxsplit=1)[0]
        by_source.setdefault(source, []).append(device)
    rng = np.random.default_rng(20260929)
    deltas = np.asarray([candidate[name] - baseline[name] for name in devices])
    delta_by_device = dict(zip(devices, deltas, strict=True))
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        source_means = []
        for source in sorted(by_source):
            members = by_source[source]
            sampled = rng.choice(members, size=len(members), replace=True)
            source_means.append(np.mean([delta_by_device[name] for name in sampled]))
        draws[index] = np.mean(source_means)
    return {
        "mean_device_paired_difference": float(np.mean(deltas)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": 20260929,
    }


def aggregate(run_root: Path, seeds: tuple[int, ...]) -> dict[str, Any]:
    device_scores: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    fold_ids: set[str] = set()
    candidate_keys: set[str] = set()
    for seed in seeds:
        seed_key = str(seed)
        device_scores[seed_key] = {}
        for fold in range(5):
            fold_dir = run_root / f"seed{seed}" / f"fold{fold}"
            manifest_path = fold_dir / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing fold manifest: {manifest_path}")
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("status") != "complete":
                raise ValueError(f"Fold is not complete: {manifest_path}")
            if manifest.get("seed") != seed or manifest.get("fold") != fold:
                raise ValueError(f"Seed/fold mismatch in {manifest_path}")
            fold_ids.update(manifest["fold_devices"]["held_out"])
            for candidate in manifest["candidates"]:
                key = f"w{candidate['window']}_s{candidate['stride']}"
                candidate_keys.add(key)
                metrics = candidate["metrics_native_rul"]["per_device"]
                expected_prediction = (
                    fold_dir
                    / f"fold{fold}_window{candidate['window']}_stride{candidate['stride']}_predictions.npz"
                )
                if not expected_prediction.exists():
                    raise FileNotFoundError(expected_prediction)
                with np.load(expected_prediction, allow_pickle=False) as result:
                    ids = np.asarray(result["unit_ids"]).astype(str)
                    predictions = np.asarray(result["predictions"], dtype=np.float64)
                    targets = np.asarray(result["targets_native"], dtype=np.float64)
                if not (len(ids) == len(predictions) == len(targets)):
                    raise ValueError(
                        f"Prediction rows are misaligned: {expected_prediction}"
                    )
                if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
                    raise ValueError(
                        f"Non-finite prediction rows: {expected_prediction}"
                    )
                if set(np.unique(ids)) != set(manifest["fold_devices"]["held_out"]):
                    raise ValueError(
                        f"Unexpected held-out devices in {expected_prediction}"
                    )
                for device, metrics_by_device in metrics.items():
                    if device not in set(np.unique(ids)):
                        raise ValueError(
                            f"Metric device absent from predictions: {device}"
                        )
                    device_scores[seed_key].setdefault(key, {})[device] = {
                        metric: float(metrics_by_device[metric]) for metric in METRICS
                    }

    expected_devices = {
        f"DS{source:02d}-unit{unit:02d}"
        for source in (1, 4, 5, 7)
        for unit in (1, 2, 3, 4, 5)
    }
    if fold_ids != expected_devices:
        raise ValueError(
            "Out-of-fold inventory is incomplete or contains extra devices"
        )
    if candidate_keys != {BASELINE, "w5_s1", "w10_s5", "w20_s5", "w50_s50"}:
        raise ValueError(f"Unexpected candidate inventory: {sorted(candidate_keys)}")

    summary: dict[str, Any] = {
        "status": "complete",
        "protocol": "NC-P grouped device-held-out temporal controls v1",
        "run_root": str(run_root.resolve()),
        "seeds": list(seeds),
        "held_out_device_count": len(expected_devices),
        "source_stratified_device_bootstrap": {
            "resampling_unit": "engine",
            "strata": ["DS01", "DS04", "DS05", "DS07"],
            "engines_per_source": 5,
            "replicates": 10_000,
            "interval": "percentile 95%",
        },
        "seed_metrics": {},
        "seed_mean_and_sd": {},
        "paired_device_bootstrap_vs_w1_s1": {},
        "per_device_seed_averaged_metrics": {},
    }
    for key in sorted(candidate_keys):
        summary["seed_metrics"][key] = {}
        for metric in METRICS:
            seed_means = []
            for seed in seeds:
                per_device = device_scores[str(seed)][key]
                vals = [scores[metric] for scores in per_device.values()]
                valid = [value for value in vals if np.isfinite(value)]
                mean = float(np.mean(valid)) if valid else float("nan")
                summary["seed_metrics"][key].setdefault(str(seed), {})[metric] = mean
                seed_means.append(mean)
            summary["seed_mean_and_sd"].setdefault(key, {})[metric] = {
                "mean_across_seeds": float(np.mean(seed_means)),
                "sd_across_seeds": float(np.std(seed_means, ddof=1)),
            }

            averaged_by_device = {
                device: float(
                    np.mean(
                        [
                            device_scores[str(seed)][key][device][metric]
                            for seed in seeds
                        ]
                    )
                )
                for device in sorted(expected_devices)
            }
            summary["per_device_seed_averaged_metrics"].setdefault(key, {})[metric] = (
                averaged_by_device
            )
            if key != BASELINE:
                baseline_by_device = {
                    device: float(
                        np.mean(
                            [
                                device_scores[str(seed)][BASELINE][device][metric]
                                for seed in seeds
                            ]
                        )
                    )
                    for device in sorted(expected_devices)
                }
                summary["paired_device_bootstrap_vs_w1_s1"].setdefault(key, {})[
                    metric
                ] = _bootstrap_device_difference(averaged_by_device, baseline_by_device)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()
    summary = aggregate(args.run_root, tuple(args.seeds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    for candidate, metrics in summary["seed_mean_and_sd"].items():
        mae = metrics["mae"]
        print(
            f"{candidate}: device-macro native MAE "
            f"{mae['mean_across_seeds']:.4f} ± {mae['sd_across_seeds']:.4f}"
        )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
