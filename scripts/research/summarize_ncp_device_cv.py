#!/usr/bin/env python3
"""Aggregate pre-registered NC-P grouped-CV manifests and device predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from picid.research.ncp_device_cv import validate_full_timeline_query_rows  # noqa: E402


SEEDS = (72, 88, 101, 666, 226688)
METRICS = (
    "mae",
    "mse",
    "rmse",
    "nasa_score",
    "5pct_mae",
    "10pct_mae",
    "20pct_mae",
)
BASELINE = "w1_s1"
EXPECTED_CANDIDATES = {BASELINE, "w5_s1", "w10_s5", "w20_s5", "w50_s50"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if seeds != SEEDS:
        raise ValueError(
            f"Formal aggregation requires the registered five seeds {SEEDS}"
        )
    device_scores: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    fold_ids: set[str] = set()
    candidate_keys: set[str] = set()
    candidate_costs: dict[str, list[dict[str, float | int]]] = {}
    fold_task_wall_seconds: dict[str, dict[str, float]] = {}
    query_signature_by_fold: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
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
            fold_task_wall_seconds.setdefault(seed_key, {})[str(fold)] = float(
                manifest.get("wall_seconds", float("nan"))
            )
            query_protocol = manifest.get("query_protocol", {})
            if query_protocol.get("rule") != (
                "every transformed row in chronological order; no test/query subsampling"
            ):
                raise ValueError(
                    f"Fold does not use complete query timelines: {manifest_path}"
                )
            query_counts = {
                device: int(metadata["count"])
                for device, metadata in query_protocol.get(
                    "indices_by_device", {}
                ).items()
            }
            if set(query_counts) != set(manifest["fold_devices"]["held_out"]):
                raise ValueError(
                    f"Query inventory differs from held-out devices: {manifest_path}"
                )
            fold_ids.update(manifest["fold_devices"]["held_out"])
            fold_candidate_keys: set[str] = set()
            reference_rows: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
            for candidate in manifest["candidates"]:
                key = f"w{candidate['window']}_s{candidate['stride']}"
                if key in fold_candidate_keys:
                    raise ValueError(f"Duplicate candidate {key} in {manifest_path}")
                fold_candidate_keys.add(key)
                candidate_keys.add(key)
                metrics = candidate["metrics_native_rul"]["per_device"]
                expected_prediction = (
                    fold_dir
                    / f"fold{fold}_window{candidate['window']}_stride{candidate['stride']}_predictions.npz"
                )
                if not expected_prediction.exists():
                    raise FileNotFoundError(expected_prediction)
                if candidate.get("prediction_sha256") != _sha256(expected_prediction):
                    raise ValueError(
                        f"Prediction digest mismatch: {expected_prediction}"
                    )
                with np.load(expected_prediction, allow_pickle=False) as result:
                    ids = np.asarray(result["unit_ids"]).astype(str)
                    predictions = np.asarray(
                        result["predictions_normalized"], dtype=np.float64
                    )
                    targets = np.asarray(result["targets_native"], dtype=np.float64)
                    times = np.asarray(result["query_times"], dtype=np.int64)
                if not (len(ids) == len(predictions) == len(targets)):
                    raise ValueError(
                        f"Prediction rows are misaligned: {expected_prediction}"
                    )
                if len(times) != len(ids):
                    raise ValueError(
                        f"Query times are misaligned: {expected_prediction}"
                    )
                if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
                    raise ValueError(
                        f"Non-finite prediction rows: {expected_prediction}"
                    )
                try:
                    validate_full_timeline_query_rows(ids, times, query_counts)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid query rows in {expected_prediction}: {exc}"
                    ) from exc
                for device, expected_count in query_counts.items():
                    if (
                        candidate.get("query_rows_by_device", {}).get(device)
                        != expected_count
                    ):
                        raise ValueError(
                            f"Manifest query count mismatch for {device}: {expected_prediction}"
                        )
                current_rows = (ids, times, targets)
                if reference_rows is None:
                    reference_rows = tuple(array.copy() for array in current_rows)
                elif not all(
                    np.array_equal(reference, current)
                    for reference, current in zip(
                        reference_rows, current_rows, strict=True
                    )
                ):
                    raise ValueError(
                        f"Candidates use different query devices, times, or labels: "
                        f"{expected_prediction}"
                    )
                for device, metrics_by_device in metrics.items():
                    if device not in set(np.unique(ids)):
                        raise ValueError(
                            f"Metric device absent from predictions: {device}"
                        )
                    row = {
                        metric: float(metrics_by_device[metric]) for metric in METRICS
                    }
                    if not all(np.isfinite(value) for value in row.values()):
                        raise ValueError(
                            f"Required metric is undefined for {device}: "
                            f"{expected_prediction}"
                        )
                    device_scores[seed_key].setdefault(key, {})[device] = row
                candidate_costs.setdefault(key, []).append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "query_rows": len(ids),
                        "data_preparation_seconds": float(
                            candidate["data_preparation_seconds"]
                        ),
                        "model_initialization_seconds": float(
                            candidate["model_initialization_seconds"]
                        ),
                        "fit_seconds": float(candidate["fit_seconds"]),
                        "predict_seconds": float(candidate["predict_seconds"]),
                        "serialization_seconds": float(
                            candidate["serialization_seconds"]
                        ),
                        "digest_seconds": float(candidate["digest_seconds"]),
                        "candidate_wall_seconds": float(
                            candidate["candidate_wall_seconds"]
                        ),
                        "query_throughput_per_second": float(
                            candidate["query_throughput_per_second"]
                        ),
                        "peak_allocated_gib": float(candidate["peak_allocated_gib"]),
                        "peak_reserved_gib": float(candidate["peak_reserved_gib"]),
                    }
                )
            if fold_candidate_keys != EXPECTED_CANDIDATES:
                raise ValueError(
                    f"Incomplete candidate inventory in {manifest_path}: "
                    f"{sorted(fold_candidate_keys)}"
                )
            if reference_rows is None:
                raise ValueError(
                    f"Fold contains no candidate predictions: {manifest_path}"
                )
            previous_rows = query_signature_by_fold.get(fold)
            if previous_rows is None:
                query_signature_by_fold[fold] = tuple(
                    array.copy() for array in reference_rows
                )
            elif not all(
                np.array_equal(reference, current)
                for reference, current in zip(
                    previous_rows, reference_rows, strict=True
                )
            ):
                raise ValueError(
                    f"Seeds do not share the same held-out query rows in fold {fold}"
                )

    expected_devices = {
        f"DS{source:02d}-unit{unit:02d}"
        for source in (1, 4, 5, 7)
        for unit in (1, 2, 3, 4, 5)
    }
    if fold_ids != expected_devices:
        raise ValueError(
            "Out-of-fold inventory is incomplete or contains extra devices"
        )
    if candidate_keys != EXPECTED_CANDIDATES:
        raise ValueError(f"Unexpected candidate inventory: {sorted(candidate_keys)}")

    summary: dict[str, Any] = {
        "status": "complete",
        "protocol": "NC-P grouped device-held-out temporal controls v2",
        "query_protocol": "all rows from each held-out device, in chronological order; no query subsampling",
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "source_sha256": {
            "summarizer": _sha256(Path(__file__).resolve()),
            "device_cv_utils": _sha256(REPO_ROOT / "picid/research/ncp_device_cv.py"),
        },
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
        "compute_cost": {
            "candidate_evaluations": sum(
                len(rows) for rows in candidate_costs.values()
            ),
            "task_wall_seconds_by_seed_fold": fold_task_wall_seconds,
            "candidates": {},
        },
    }
    summary["compute_cost"]["task_wall_seconds_sum"] = float(
        sum(
            value
            for folds in fold_task_wall_seconds.values()
            for value in folds.values()
        )
    )
    for key, rows in candidate_costs.items():
        summary["compute_cost"]["candidates"][key] = {
            "evaluation_count": len(rows),
            "total_query_rows": int(sum(row["query_rows"] for row in rows)),
            "mean_data_preparation_seconds": float(
                np.mean([row["data_preparation_seconds"] for row in rows])
            ),
            "mean_model_initialization_seconds": float(
                np.mean([row["model_initialization_seconds"] for row in rows])
            ),
            "mean_fit_seconds": float(np.mean([row["fit_seconds"] for row in rows])),
            "mean_predict_seconds": float(
                np.mean([row["predict_seconds"] for row in rows])
            ),
            "sum_fit_predict_seconds": float(
                sum(row["fit_seconds"] + row["predict_seconds"] for row in rows)
            ),
            "mean_candidate_wall_seconds": float(
                np.mean([row["candidate_wall_seconds"] for row in rows])
            ),
            "total_serialization_seconds": float(
                sum(row["serialization_seconds"] for row in rows)
            ),
            "query_throughput_rows_per_second_weighted": float(
                sum(row["query_rows"] for row in rows)
                / sum(row["predict_seconds"] for row in rows)
            ),
            "max_peak_allocated_gib": float(
                max(row["peak_allocated_gib"] for row in rows)
            ),
            "max_peak_reserved_gib": float(
                max(row["peak_reserved_gib"] for row in rows)
            ),
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
