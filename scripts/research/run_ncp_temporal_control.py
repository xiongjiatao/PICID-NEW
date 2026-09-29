#!/usr/bin/env python3
"""Run seed-fixed, device-isolated temporal representation controls on NC-P."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from picid.research.temporal_controls import (  # noqa: E402
    contiguous_unit_slices,
    history_windows,
    multiscale_summary,
    regression_metrics_by_device,
    stratified_timeline_indices,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _constant_rul_scale(preprocessing_path: Path) -> float:
    """Read the canonical constant RUL target scale from the exported config."""
    from omegaconf import OmegaConf

    config = OmegaConf.load(preprocessing_path)
    scaler = config.transforms.scaler_rul.transform
    scaler_name = str(scaler.get("_target_", ""))
    if not scaler_name.endswith(".ConstantScaler"):
        raise ValueError(
            f"Expected the frozen ConstantScaler for RUL, got {scaler_name!r}"
        )
    factor = float(scaler.factor)
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError(f"Invalid constant RUL target scale: {factor}")
    return factor


def _snapshot_gpu_state() -> dict[str, str]:
    try:
        gpu_state = subprocess.run(
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
        process_state = subprocess.run(
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
        return {"gpu_memory_utilization": gpu_state, "compute_processes": process_state}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _device_name(row: np.ndarray) -> str:
    source, unit = (int(value) for value in row)
    return f"DS{source:02d}-unit{unit:02d}"


def _load_units(
    input_dir: Path, split: str
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    features = np.load(input_dir / f"{split}_features.npy", mmap_mode="r")
    targets = np.load(input_dir / f"{split}_rul.npy", mmap_mode="r").reshape(-1)
    unit_ids = np.load(input_dir / f"{split}_unit_id.npy", mmap_mode="r")
    if len(features) != len(targets) or len(features) != len(unit_ids):
        raise ValueError(f"{split} arrays have inconsistent row counts")
    groups = contiguous_unit_slices(unit_ids)
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for _, row_slice in groups:
        name = _device_name(unit_ids[row_slice.start])
        if name in result:
            raise ValueError(
                f"Duplicate device identity after source/unit mapping: {name}"
            )
        result[name] = (
            np.asarray(features[row_slice], dtype=np.float32),
            np.asarray(targets[row_slice], dtype=np.float32),
        )
    return result


def _prepare_context_data(
    units: dict[str, tuple[np.ndarray, np.ndarray]],
    window: int,
    context_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, list[int]]]:
    names = sorted(units)
    if context_rows < len(names):
        raise ValueError("context_rows must allocate at least one row per device")
    base, remainder = divmod(context_rows, len(names))
    feature_rows, target_rows, device_rows = [], [], []
    selected_by_device: dict[str, list[int]] = {}
    for rank, name in enumerate(names):
        features, targets = units[name]
        count = base + int(rank < remainder)
        selected = stratified_timeline_indices(len(features), count, seed + rank)
        feature_rows.append(history_windows(features, selected, window))
        target_rows.append(targets[selected])
        device_rows.append(np.full(count, name, dtype=object))
        selected_by_device[name] = selected.tolist()
    return (
        np.concatenate(feature_rows),
        np.concatenate(target_rows),
        np.concatenate(device_rows),
        names,
        selected_by_device,
    )


def _prepare_query_data(
    units: dict[str, tuple[np.ndarray, np.ndarray]],
    window: int,
    queries_per_device: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_x, all_y, all_ids, all_times = [], [], [], []
    for name, (features, targets) in sorted(units.items()):
        indices = np.unique(
            np.linspace(
                0,
                len(features) - 1,
                min(queries_per_device, len(features)),
                dtype=np.int64,
            )
        )
        all_x.append(history_windows(features, indices, window))
        all_y.append(targets[indices])
        all_ids.append(np.full(len(indices), name, dtype=object))
        all_times.append(indices)
    return (
        np.concatenate(all_x),
        np.concatenate(all_y),
        np.concatenate(all_ids),
        np.concatenate(all_times),
    )


def _manifest_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--context-rows", type=int, default=2048)
    parser.add_argument("--queries-per-device", type=int, default=1024)
    parser.add_argument("--n-ensembles", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=512)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--weight-path", type=Path, required=True)
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(args.physical_gpu):
        raise RuntimeError(
            "Set CUDA_VISIBLE_DEVICES to exactly the requested physical GPU "
            f"({args.physical_gpu}); got {visible!r}"
        )
    if args.physical_gpu not in range(1, 7):
        raise ValueError(
            "This study was explicitly authorized on physical GPUs 1 through 6"
        )

    import torch
    from importlib.metadata import version
    from tabdpt import TabDPTRegressor

    if version("tabdpt") != "1.3.0":
        raise RuntimeError(f"Expected tabdpt==1.3.0, got {version('tabdpt')}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("A single visible CUDA device is required")
    torch.cuda.set_device(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    expected_peak_gib = 8.0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    manifest_path = output_dir / "manifest.json"
    input_manifest_path = args.input_dir / "manifest.json"
    preprocessing_path = args.input_dir / "preprocessing.yaml"
    input_manifest = json.loads(input_manifest_path.read_text())
    target_scale = _constant_rul_scale(preprocessing_path)
    input_hashes = {
        path.name: _sha256(path)
        for path in [
            input_manifest_path,
            preprocessing_path,
            args.input_dir / "train_features.npy",
            args.input_dir / "train_rul.npy",
            args.input_dir / "train_unit_id.npy",
            args.input_dir / "val_features.npy",
            args.input_dir / "val_rul.npy",
            args.input_dir / "val_unit_id.npy",
        ]
    }
    manifest: dict[str, Any] = {
        "status": "running",
        "command": shlex.join(sys.argv),
        "cwd": os.getcwd(),
        "source_root": str(root),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "source_sha256": {
            "script": _sha256(Path(__file__).resolve()),
            "temporal_controls": _sha256(root / "picid/research/temporal_controls.py"),
        },
        "data_manifest": str(input_manifest_path.resolve()),
        "data_file_sha256": input_hashes,
        "dataset_export_manifest": input_manifest,
        "target_scaling": {
            "transform": "ConstantScaler",
            "normalized_target_factor": target_scale,
            "native_rul_per_normalized_unit": 1.0 / target_scale,
            "source": str(preprocessing_path.resolve()),
        },
        "physical_gpu": args.physical_gpu,
        "cuda_visible_devices": visible,
        "logical_device": "cuda:0",
        "process_id": os.getpid(),
        "gpu_memory_preflight": {
            "free_gib": free_bytes / 1024**3,
            "total_gib": total_bytes / 1024**3,
            "conservative_expected_peak_gib": expected_peak_gib,
        },
        "gpu_snapshot_before_start": _snapshot_gpu_state(),
        "seed": args.seed,
        "model": {
            "name": "TabDPT",
            "version": version("tabdpt"),
            "weight_path": str(args.weight_path.resolve()),
            "weight_sha256": _sha256(args.weight_path),
            "feature_reduction": "pca",
            "context_reduction": "subsample",
            "context_rows": args.context_rows,
            "n_ensembles": args.n_ensembles,
            "inference_batch_size": args.inference_batch_size,
            "compile": False,
        },
        "parameters": vars(args)
        | {
            "input_dir": str(args.input_dir.resolve()),
            "output_dir": str(output_dir),
            "weight_path": str(args.weight_path.resolve()),
        },
        "representations": [],
    }
    _manifest_write(manifest_path, manifest)
    if free_bytes < expected_peak_gib * 1024**3:
        manifest["status"] = "failed_resource_preflight"
        _manifest_write(manifest_path, manifest)
        raise RuntimeError(
            f"Only {free_bytes / 1024**3:.2f} GiB free; need conservative "
            f"{expected_peak_gib:.1f} GiB for this model configuration"
        )

    started = time.perf_counter()
    train_units = _load_units(args.input_dir, "train")
    val_units = _load_units(args.input_dir, "val")
    if len(train_units) != 20 or len(val_units) != 4:
        raise ValueError(
            f"Expected NC-P 20 train/4 validation devices, got {len(train_units)}/{len(val_units)}"
        )
    context_cache: dict[
        int,
        tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, list[int]]],
    ] = {}
    query_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    window_preparation_seconds: dict[int, float] = {}
    for window in (1, 10, 50):
        window_started = time.perf_counter()
        context_cache[window] = _prepare_context_data(
            train_units, window, args.context_rows, args.seed
        )
        query_cache[window] = _prepare_query_data(
            val_units, window, args.queries_per_device
        )
        window_preparation_seconds[window] = time.perf_counter() - window_started
    manifest["window_preparation_seconds_by_width"] = window_preparation_seconds
    _manifest_write(manifest_path, manifest)

    if args.preflight_only:
        context_x, context_y, _, _, _ = context_cache[50]
        query_x, query_y, _, _ = query_cache[50]
        preflight_count = min(256, len(query_y))
        fit_x = context_x.reshape(len(context_x), -1).astype(np.float32)
        test_x = (
            query_x[:preflight_count].reshape(preflight_count, -1).astype(np.float32)
        )
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats(0)
        model = TabDPTRegressor(
            device="cuda",
            model_weight_path=str(args.weight_path.resolve()),
            feature_reduction="pca",
            context_reduction="subsample",
            compile=False,
        )
        fit_started = time.perf_counter()
        model.fit(fit_x, context_y)
        torch.cuda.synchronize()
        fit_seconds = time.perf_counter() - fit_started
        predict_started = time.perf_counter()
        pilot = model.predict(
            test_x,
            output_type="mean",
            seed=args.seed,
            context_size=args.context_rows,
            n_ensembles=args.n_ensembles,
            batch_size=args.inference_batch_size,
        )
        torch.cuda.synchronize()
        predict_seconds = time.perf_counter() - predict_started
        manifest["status"] = "preflight_complete"
        manifest["preflight"] = {
            "representation": "raw_window_50",
            "pilot_queries": int(preflight_count),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "query_throughput_per_second": float(preflight_count / predict_seconds),
            "peak_allocated_gib": float(torch.cuda.max_memory_allocated(0) / 1024**3),
            "peak_reserved_gib": float(torch.cuda.max_memory_reserved(0) / 1024**3),
            "conservative_full_study_runtime_estimate_seconds": float(
                5 * (fit_seconds + len(query_y) / preflight_count * predict_seconds)
                + sum(window_preparation_seconds.values())
            ),
            "estimate_includes": [
                "five model fits and predictions extrapolated from the W50 pilot",
                "construction of all three causal window widths",
            ],
            "estimate_excludes": [
                "PCA and multiscale transform time, which is measured per representation in a full run",
                "cached dataset preprocessing",
            ],
            "prediction_finite": bool(np.isfinite(np.asarray(pilot)).all()),
            "query_rows_per_device": {
                str(unit): int(np.sum(query_cache[50][2] == unit))
                for unit in np.unique(query_cache[50][2])
            },
        }
        manifest["wall_seconds"] = time.perf_counter() - started
        manifest["gpu_snapshot_after_finish"] = _snapshot_gpu_state()
        _manifest_write(manifest_path, manifest)
        return 0

    horizons_payload_path = (
        args.input_dir.parents[1] / "results" / "nc_p_critical_horizons.json"
    )
    if horizons_payload_path.exists():
        horizon_payload = json.loads(horizons_payload_path.read_text())
        horizons = {
            key: float(value["rul_threshold"])
            for key, value in horizon_payload["horizons"].items()
        }
        manifest["critical_horizon_source"] = str(horizons_payload_path.resolve())
    else:
        horizons = {"5pct": 4.2, "10pct": 8.4, "20pct": 16.8}
        manifest["critical_horizon_source"] = (
            "frozen NC-P horizon in PICID_REPRODUCTION.md (median train-device max RUL 84)"
        )
    manifest["critical_horizons_native_units"] = horizons

    representations = [
        ("raw_window_1", 1, "raw"),
        ("raw_window_10", 10, "raw"),
        ("raw_window_50", 50, "raw"),
        ("train_pca_window_50_128", 50, "pca128"),
        ("fixed_multiscale_window_50", 50, "multiscale"),
    ]

    for name, window, mode in representations:
        rep_start = time.perf_counter()
        context_x, context_y, context_ids, context_devices, context_indices = (
            context_cache[window]
        )
        query_x, query_y, query_ids, query_times = query_cache[window]
        query_native_rul = query_y.astype(np.float64) / target_scale
        if mode == "pca128":
            from sklearn.decomposition import PCA

            flat_train = context_x.reshape(len(context_x), -1)
            flat_query = query_x.reshape(len(query_x), -1)
            pca = PCA(
                n_components=min(128, len(flat_train) - 1, flat_train.shape[1]),
                random_state=args.seed,
            )
            pca.fit(flat_train)
            fit_x = pca.transform(flat_train).astype(np.float32)
            test_x = pca.transform(flat_query).astype(np.float32)
            representation_meta = {
                "projection": "sklearn.PCA(full_solver)",
                "components": int(pca.n_components_),
                "fit_rows": int(len(flat_train)),
                "fit_scope": "selected train contexts only",
                "explained_variance_ratio_sum": float(
                    pca.explained_variance_ratio_.sum()
                ),
            }
        elif mode == "multiscale":
            fit_x = multiscale_summary(context_x)
            test_x = multiscale_summary(query_x)
            representation_meta = {
                "statistics": ["mean", "standard_deviation", "linear_slope"],
                "suffix_scales": [1, 5, 10, 25, 50],
                "feature_count": int(fit_x.shape[1]),
            }
        else:
            fit_x = context_x.reshape(len(context_x), -1)
            test_x = query_x.reshape(len(query_x), -1)
            representation_meta = {"feature_count": int(fit_x.shape[1])}
        representation_transform_seconds = time.perf_counter() - rep_start

        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats(0)
        model = TabDPTRegressor(
            device="cuda",
            model_weight_path=str(args.weight_path.resolve()),
            feature_reduction="pca",
            context_reduction="subsample",
            compile=False,
        )
        fit_started = time.perf_counter()
        model.fit(
            np.asarray(fit_x, dtype=np.float32), np.asarray(context_y, dtype=np.float32)
        )
        torch.cuda.synchronize()
        fit_seconds = time.perf_counter() - fit_started
        predict_started = time.perf_counter()
        predictions = model.predict(
            np.asarray(test_x, dtype=np.float32),
            output_type="mean",
            seed=args.seed,
            context_size=args.context_rows,
            n_ensembles=args.n_ensembles,
            batch_size=args.inference_batch_size,
        )
        torch.cuda.synchronize()
        predict_seconds = time.perf_counter() - predict_started
        predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
        normalized_metrics = regression_metrics_by_device(
            predictions,
            query_y,
            query_ids,
            horizons,
            critical_values=query_native_rul,
        )
        native_rul_metrics = regression_metrics_by_device(
            predictions / target_scale,
            query_y / target_scale,
            query_ids,
            horizons,
            critical_values=query_native_rul,
        )
        metrics = {
            "normalized_target": normalized_metrics,
            "native_rul": native_rul_metrics,
        }
        result = {
            "name": name,
            "window": window,
            "representation": mode,
            "seed": args.seed,
            "context_devices": context_devices,
            "context_endpoint_indices_by_device": context_indices,
            "context_rows_by_device": {
                k: int(np.sum(context_ids == k)) for k in context_devices
            },
            "query_rows_by_device": {
                str(k): int(np.sum(query_ids == k)) for k in np.unique(query_ids)
            },
            "input_dimension": int(fit_x.shape[1]),
            "representation_details": representation_meta,
            "window_preparation_seconds": window_preparation_seconds[window],
            "representation_transform_seconds": representation_transform_seconds,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "total_candidate_seconds": time.perf_counter() - rep_start,
            "end_to_end_candidate_seconds": float(
                window_preparation_seconds[window] + time.perf_counter() - rep_start
            ),
            "query_throughput_per_second": float(len(query_y) / predict_seconds),
            "peak_allocated_gib": float(torch.cuda.max_memory_allocated(0) / 1024**3),
            "peak_reserved_gib": float(torch.cuda.max_memory_reserved(0) / 1024**3),
            "metrics": metrics,
        }
        np.savez_compressed(
            output_dir / f"{name}_seed{args.seed}_predictions.npz",
            predictions=predictions,
            targets=query_y,
            raw_rul_native=query_native_rul,
            unit_ids=query_ids,
            query_times=query_times,
        )
        manifest["representations"].append(result)
        _manifest_write(manifest_path, manifest)
        del model, fit_x, test_x, predictions
        gc.collect()
        torch.cuda.empty_cache()

    manifest["status"] = "complete"
    manifest["wall_seconds"] = time.perf_counter() - started
    manifest["gpu_snapshot_after_finish"] = _snapshot_gpu_state()
    _manifest_write(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        output_arg = next(
            (i + 1 for i, arg in enumerate(sys.argv[:-1]) if arg == "--output-dir"),
            None,
        )
        if output_arg is not None:
            target = Path(sys.argv[output_arg]) / "manifest.json"
            if target.exists():
                try:
                    payload = json.loads(target.read_text())
                    payload["status"] = "failed"
                    payload["error"] = f"{type(exc).__name__}: {exc}"
                    _manifest_write(target, payload)
                except Exception:
                    pass
        raise
    raise SystemExit(exit_code)
