#!/usr/bin/env python3
"""Run one source-stratified NC-P device-held-out fold for temporal controls."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from picid.research.ncp_device_cv import (  # noqa: E402
    balanced_strided_context_indices,
    full_timeline_query_indices,
    fold_critical_horizons,
    nc_device_names,
    source_stratified_device_folds,
)
from picid.research.temporal_controls import (  # noqa: E402
    contiguous_unit_slices,
    history_windows,
    nasa_score_by_device,
    regression_metrics_by_device,
)


PAPER_WINDOW_STRIDE_GRID = ((1, 1), (5, 1), (10, 5), (20, 5), (50, 50))
PAPER_SEEDS = (72, 88, 101, 666, 226688)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_units(
    input_dir: Path,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, np.ndarray
]:
    features = np.load(input_dir / "train_features.npy", mmap_mode="r")
    targets = np.load(input_dir / "train_rul.npy", mmap_mode="r").reshape(-1)
    unit_ids = np.load(input_dir / "train_unit_id.npy", mmap_mode="r")
    if not (len(features) == len(targets) == len(unit_ids)):
        raise ValueError("NC-P train arrays have inconsistent row counts")
    names = nc_device_names(unit_ids)
    groups = contiguous_unit_slices(unit_ids)
    units: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for _, row_slice in groups:
        name = str(names[row_slice.start])
        if name in units:
            raise ValueError(f"Device rows are not unique and contiguous: {name}")
        units[name] = (
            np.asarray(features[row_slice], dtype=np.float32),
            np.asarray(targets[row_slice], dtype=np.float32),
        )
    return units, np.asarray(targets), np.asarray(unit_ids), names


def _gpu_snapshot() -> dict[str, str]:
    try:
        gpus = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        processes = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return {"gpu_memory_utilization": gpus, "compute_processes": processes}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, choices=PAPER_SEEDS, required=True)
    parser.add_argument("--context-rows", type=int, default=2048)
    parser.add_argument("--n-ensembles", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=512)
    parser.add_argument("--weight-path", type=Path, required=True)
    parser.add_argument("--concurrent-tasks", type=str, default="")
    args = parser.parse_args()

    if (args.context_rows, args.n_ensembles, args.inference_batch_size) != (
        2048,
        8,
        512,
    ):
        raise ValueError(
            "Registered NC-P device-CV budget is context=2048, ensembles=8, "
            "inference_batch_size=512"
        )

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(args.physical_gpu):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES must be exactly physical GPU {args.physical_gpu}; "
            f"got {visible!r}"
        )
    if args.physical_gpu not in range(1, 7):
        raise ValueError("This study is authorized on physical GPUs 1 through 6")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("status") == "complete":
            raise FileExistsError(
                f"Refusing to overwrite completed run {manifest_path}"
            )
        archive_dir = output_dir / "attempts"
        archive_dir.mkdir(exist_ok=True)
        archive_path = archive_dir / f"previous_attempt_{time.time_ns()}.json"
        shutil.copy2(manifest_path, archive_path)
    _write_manifest(
        manifest_path,
        {
            "status": "starting",
            "command": f"CUDA_VISIBLE_DEVICES={visible} {shlex.join(sys.argv)}",
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "physical_gpu": args.physical_gpu,
            "cuda_visible_devices": visible,
            "logical_device": "cuda:0",
            "concurrent_tasks": args.concurrent_tasks,
            "seed": args.seed,
            "fold": args.fold,
            "started_unix_ns": time.time_ns(),
        },
    )

    import torch
    from importlib.metadata import version
    from omegaconf import OmegaConf
    from tabdpt import TabDPTRegressor

    if version("tabdpt") != "1.3.0":
        raise RuntimeError(f"Expected tabdpt==1.3.0, got {version('tabdpt')}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA device is required")
    torch.cuda.set_device(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    expected_peak_gib = 8.0

    input_dir = args.input_dir.resolve()
    weight_path = args.weight_path.resolve()
    input_manifest_path = input_dir / "manifest.json"
    preprocessing_path = input_dir / "preprocessing.yaml"
    input_manifest = json.loads(input_manifest_path.read_text())
    preprocessing = OmegaConf.load(preprocessing_path)
    target_scale = float(preprocessing.transforms.scaler_rul.transform.factor)
    aggregation = preprocessing.transforms.subsample_features_target.transform
    if input_manifest.get("dataset") != "nc_p":
        raise ValueError("This runner requires the NC-P dataset manifest")
    if (input_manifest.get("window"), input_manifest.get("stride")) != (1, 1):
        raise ValueError("Input cache must contain unwindowed, stride-1 NC-P rows")
    if (
        int(aggregation.step),
        int(aggregation.window_size),
        str(aggregation.aggregation),
    ) != (60, 60, "mean"):
        raise ValueError(
            "Input cache does not use the registered 60/60 mean aggregation"
        )
    if not np.isclose(target_scale, 0.01, rtol=0.0, atol=1e-12):
        raise ValueError(
            "Input cache does not use the registered NC-P RUL ×0.01 target"
        )
    input_paths = [
        input_manifest_path,
        preprocessing_path,
        input_dir / "train_features.npy",
        input_dir / "train_rul.npy",
        input_dir / "train_unit_id.npy",
    ]
    train_data_manifest = input_manifest["splits"]["train"]
    expected_array_hashes = {
        "train_features.npy": train_data_manifest["features"]["sha256"],
        "train_rul.npy": train_data_manifest["rul"]["sha256"],
        "train_unit_id.npy": train_data_manifest["unit_id"]["sha256"],
    }
    for path in input_paths:
        expected_hash = expected_array_hashes.get(path.name)
        if expected_hash is not None and _sha256(path) != expected_hash:
            raise ValueError(f"Input array does not match its manifest digest: {path}")
    manifest: dict[str, Any] = {
        "status": "running",
        "protocol": "NC-P grouped device-held-out temporal controls v2",
        "protocol_scope": "supplementary development-pool CV; not the paper's canonical split or a blind test",
        "target_paper": "https://arxiv.org/html/2606.05481v1",
        "command": f"CUDA_VISIBLE_DEVICES={visible} {shlex.join(sys.argv)}",
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "source_root": str(REPO_ROOT),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "source_sha256": {
            "runner": _sha256(Path(__file__).resolve()),
            "fold_utils": _sha256(REPO_ROOT / "picid/research/ncp_device_cv.py"),
            "temporal_controls": _sha256(
                REPO_ROOT / "picid/research/temporal_controls.py"
            ),
        },
        "input_manifest": str(input_manifest_path),
        "input_manifest_sha256": _sha256(input_manifest_path),
        "input_files_sha256": {path.name: _sha256(path) for path in input_paths},
        "input_dataset_manifest": input_manifest,
        "data_protocol": {
            "dataset": "N-CMAPSS NC-P development training pool",
            "source_unit_identity": "(n_DS, source-local unit)",
            "raw_feature_aggregation": "non-overlapping 60/60 mean",
            "target": "RUL multiplied by 0.01 in cached normalized space",
            "scaler_note": "fixed PICID benchmark scaler; provenance sensitivity remains required",
            "input_window": 1,
            "input_stride": 1,
        },
        "physical_gpu": args.physical_gpu,
        "cuda_visible_devices": visible,
        "logical_device": "cuda:0",
        "process_id": os.getpid(),
        "concurrent_tasks": args.concurrent_tasks,
        "gpu_snapshot_before_start": _gpu_snapshot(),
        "gpu_memory_preflight": {
            "free_gib": free_bytes / 1024**3,
            "total_gib": total_bytes / 1024**3,
            "expected_peak_gib": expected_peak_gib,
        },
        "seed": args.seed,
        "fold": args.fold,
        "fold_assignment": "hold out the same source-local unit 1..5 across DS01/04/05/07",
        "target_scale": {
            "normalized_factor": target_scale,
            "native_rul_per_normalized_unit": 1.0 / target_scale,
        },
        "model": {
            "name": "TabDPT",
            "version": version("tabdpt"),
            "weight_path": str(weight_path),
            "weight_sha256": _sha256(weight_path),
            "feature_reduction": "pca",
            "context_reduction": "subsample",
            "context_rows": args.context_rows,
            "n_ensembles": args.n_ensembles,
            "inference_batch_size": args.inference_batch_size,
            "compile": False,
        },
        "parameters": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "physical_gpu": args.physical_gpu,
            "fold": args.fold,
            "seed": args.seed,
            "context_rows": args.context_rows,
            "n_ensembles": args.n_ensembles,
            "inference_batch_size": args.inference_batch_size,
            "weight_path": str(weight_path),
        },
        "candidate_grid": [
            {"window": window, "stride": stride}
            for window, stride in PAPER_WINDOW_STRIDE_GRID
        ],
        "progress": {
            "completed_candidates": 0,
            "total_candidates": len(PAPER_WINDOW_STRIDE_GRID),
            "estimated_remaining_seconds": None,
        },
        "candidates": [],
    }
    _write_manifest(manifest_path, manifest)
    if free_bytes < expected_peak_gib * 1024**3:
        manifest["status"] = "failed_resource_preflight"
        _write_manifest(manifest_path, manifest)
        raise RuntimeError(
            f"Only {free_bytes / 1024**3:.2f} GiB free, below the "
            f"{expected_peak_gib:.1f} GiB conservative requirement"
        )

    started = time.perf_counter()
    train_units, all_targets, all_unit_ids, _ = _read_units(input_dir)
    folds = source_stratified_device_folds(all_unit_ids)
    fold = folds[args.fold]
    train_devices = list(fold["train_devices"])
    test_devices = list(fold["test_devices"])
    if len(train_devices) != 16 or len(test_devices) != 4:
        raise ValueError("Fold must contain 16 train and four held-out engines")
    heldout_fold = args.fold
    horizons = fold_critical_horizons(
        all_targets[np.asarray(fold["train_mask"])],
        all_unit_ids[np.asarray(fold["train_mask"])],
        native_units_per_target_unit=1.0 / target_scale,
    )
    manifest["fold_devices"] = {
        "train": train_devices,
        "held_out": test_devices,
        "horizons_native_rul": horizons,
        "critical_horizon_source": "median maximum RUL across 16 fold-training engines only",
    }
    _write_manifest(manifest_path, manifest)

    if len(set(train_devices) & set(test_devices)):
        raise ValueError("Device identity leaked across fold roles")
    train_fold_units = {name: train_units[name] for name in train_devices}
    query_indices = {
        name: full_timeline_query_indices(len(train_units[name][0]))
        for name in test_devices
    }
    target_rows: list[np.ndarray] = []
    query_ids: list[np.ndarray] = []
    query_times: list[np.ndarray] = []
    for name in test_devices:
        features, targets = train_units[name]
        indices = query_indices[name]
        target_rows.append(targets[indices])
        query_ids.append(np.full(len(indices), name, dtype=object))
        query_times.append(indices)
    manifest["query_protocol"] = {
        "rule": "every transformed row in chronological order; no test/query subsampling",
        "indices_by_device": {
            name: {
                "count": int(len(indices)),
                "first": int(indices[0]),
                "last": int(indices[-1]),
                "sha256_little_endian_int64": hashlib.sha256(
                    indices.astype("<i8", copy=False).tobytes()
                ).hexdigest(),
            }
            for name, indices in query_indices.items()
        },
        "total_rows": int(sum(len(indices) for indices in query_indices.values())),
    }
    _write_manifest(manifest_path, manifest)

    candidate_wall_times: list[float] = []
    candidate_total = len(PAPER_WINDOW_STRIDE_GRID)
    for candidate_index, (window, stride) in enumerate(
        PAPER_WINDOW_STRIDE_GRID, start=1
    ):
        candidate_started = time.perf_counter()
        print(
            f"fold={heldout_fold} seed={args.seed} window={window} stride={stride} "
            f"candidate={candidate_index}/{candidate_total} status=preparing full-query evaluation",
            flush=True,
        )
        preparation_started = time.perf_counter()
        context_indices = balanced_strided_context_indices(
            train_fold_units,
            stride=stride,
            context_rows=args.context_rows,
            seed=args.seed + 1009 * args.fold + 37 * stride,
        )
        context_x_parts: list[np.ndarray] = []
        context_y_parts: list[np.ndarray] = []
        for name in train_devices:
            features, targets = train_fold_units[name]
            endpoints = context_indices[name]
            context_x_parts.append(history_windows(features, endpoints, window))
            context_y_parts.append(targets[endpoints])
        context_x = np.concatenate(context_x_parts)
        context_y = np.concatenate(context_y_parts)

        query_x_parts: list[np.ndarray] = []
        for name in test_devices:
            features, _ = train_units[name]
            query_x_parts.append(history_windows(features, query_indices[name], window))
        query_x = np.concatenate(query_x_parts)
        query_y = np.concatenate(target_rows)
        ids = np.concatenate(query_ids)
        times = np.concatenate(query_times)
        fit_x = context_x.reshape(len(context_x), -1).astype(np.float32, copy=False)
        predict_x = query_x.reshape(len(query_x), -1).astype(np.float32, copy=False)
        preparation_seconds = time.perf_counter() - preparation_started

        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats(0)
        model_init_started = time.perf_counter()
        model = TabDPTRegressor(
            device="cuda",
            model_weight_path=str(weight_path),
            feature_reduction="pca",
            context_reduction="subsample",
            compile=False,
        )
        torch.cuda.synchronize()
        model_initialization_seconds = time.perf_counter() - model_init_started
        fit_started = time.perf_counter()
        model.fit(fit_x, context_y.astype(np.float32, copy=False))
        torch.cuda.synchronize()
        fit_seconds = time.perf_counter() - fit_started
        predict_started = time.perf_counter()
        predictions = model.predict(
            predict_x,
            output_type="mean",
            seed=args.seed,
            context_size=args.context_rows,
            n_ensembles=args.n_ensembles,
            batch_size=args.inference_batch_size,
        )
        torch.cuda.synchronize()
        predict_seconds = time.perf_counter() - predict_started
        predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
        native_target = query_y.astype(np.float64) / target_scale
        normalized_metrics = regression_metrics_by_device(
            predictions,
            query_y,
            ids,
            horizons,
            critical_values=native_target,
        )
        native_metrics = regression_metrics_by_device(
            predictions / target_scale,
            native_target,
            ids,
            horizons,
            critical_values=native_target,
        )
        nasa_metrics = nasa_score_by_device(
            predictions / target_scale, native_target, ids
        )
        native_metrics["device_macro_nasa_score"] = nasa_metrics["device_macro"]
        native_metrics["query_weighted_nasa_score"] = nasa_metrics["query_weighted"]
        for name, score in nasa_metrics["per_device"].items():
            native_metrics["per_device"][name]["nasa_score"] = score
        candidate = {
            "window": window,
            "stride": stride,
            "seed": args.seed,
            "context_rows_actual": int(len(context_y)),
            "context_rows_by_device": {
                name: int(len(context_indices[name])) for name in train_devices
            },
            "context_allocation": "max-min water-fill under per-device stride-eligible endpoint caps",
            "context_endpoint_indices_by_device": {
                name: context_indices[name].tolist() for name in train_devices
            },
            "query_rows_by_device": {
                name: int(len(query_indices[name])) for name in test_devices
            },
            "input_dimension": int(fit_x.shape[1]),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "query_throughput_per_second": float(len(query_y) / predict_seconds),
            "data_preparation_seconds": preparation_seconds,
            "model_initialization_seconds": model_initialization_seconds,
            "peak_allocated_gib": float(torch.cuda.max_memory_allocated(0) / 1024**3),
            "peak_reserved_gib": float(torch.cuda.max_memory_reserved(0) / 1024**3),
            "metrics_normalized_target": normalized_metrics,
            "metrics_native_rul": native_metrics,
            "nasa_score_native_rul": nasa_metrics,
            "prediction_finite": bool(np.isfinite(predictions).all()),
        }
        if not candidate["prediction_finite"]:
            raise ValueError(
                f"Non-finite predictions for window={window}, stride={stride}"
            )
        prediction_path = (
            output_dir
            / f"fold{heldout_fold}_window{window}_stride{stride}_predictions.npz"
        )
        serialization_started = time.perf_counter()
        np.savez_compressed(
            prediction_path,
            predictions_normalized=predictions,
            targets_normalized=query_y,
            targets_native=native_target,
            unit_ids=ids.astype(str),
            query_times=times,
        )
        candidate["serialization_seconds"] = time.perf_counter() - serialization_started
        candidate["prediction_file"] = prediction_path.name
        digest_started = time.perf_counter()
        candidate["prediction_sha256"] = _sha256(prediction_path)
        candidate["digest_seconds"] = time.perf_counter() - digest_started
        candidate["candidate_wall_seconds"] = time.perf_counter() - candidate_started
        manifest["candidates"].append(candidate)
        candidate_wall_times.append(float(candidate["candidate_wall_seconds"]))
        remaining_candidates = candidate_total - len(candidate_wall_times)
        eta_seconds = (
            float(np.mean(candidate_wall_times) * remaining_candidates)
            if remaining_candidates
            else 0.0
        )
        manifest["progress"] = {
            "completed_candidates": len(candidate_wall_times),
            "total_candidates": candidate_total,
            "estimated_remaining_seconds": eta_seconds,
        }
        _write_manifest(manifest_path, manifest)
        print(
            f"fold={heldout_fold} seed={args.seed} window={window} stride={stride} "
            f"candidate={candidate_index}/{candidate_total} "
            f"query_rows={len(query_y)} fit_s={fit_seconds:.1f} "
            f"predict_s={predict_seconds:.1f} "
            f"device_macro_native_mae={native_metrics['device_macro_mae']:.4f} "
            f"device_macro_nasa={native_metrics['device_macro_nasa_score']:.4f} "
            f"throughput_rows_s={candidate['query_throughput_per_second']:.1f} "
            f"peak_allocated_gib={candidate['peak_allocated_gib']:.2f} "
            f"eta_seconds={eta_seconds:.0f}",
            flush=True,
        )
        del model, context_x, query_x, fit_x, predict_x, predictions
        gc.collect()
        torch.cuda.empty_cache()

    manifest["status"] = "complete"
    manifest["wall_seconds"] = time.perf_counter() - started
    manifest["gpu_snapshot_after_finish"] = _gpu_snapshot()
    _write_manifest(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        output_arg = next(
            (i + 1 for i, arg in enumerate(sys.argv[:-1]) if arg == "--output-dir"),
            None,
        )
        if output_arg is not None:
            manifest_path = Path(sys.argv[output_arg]) / "manifest.json"
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text())
                    if manifest.get("status") != "complete":
                        manifest["status"] = "failed"
                        manifest["error"] = f"{type(exc).__name__}: {exc}"
                        _write_manifest(manifest_path, manifest)
                except Exception:
                    pass
        raise
