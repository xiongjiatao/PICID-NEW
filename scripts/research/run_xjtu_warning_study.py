#!/usr/bin/env python3
"""Evaluate calibrated warning probabilities for the frozen XJTU PHMD split.

The model predicts normalized HI. Warning labels use the PHMD acquisition
timeline, while model scores use only normalized HI predictions and training
assets; test-bearing total life is never used to transform a forecast.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from picid.research.warning_metrics import (  # noqa: E402
    alarm_metrics,
    apply_platt_calibrator,
    fit_platt_calibrator,
    probability_metrics,
    select_empirical_fpr_threshold,
    warning_labels_from_rul,
)
from picid.research.warning_probabilities import (  # noqa: E402
    residual_ecdf_score,
    tabdpt_distribution_cdf,
    tabdpt_distribution_mean,
)
from picid.research.prediction_io import load_normalized_predictions  # noqa: E402
from picid.research.temporal_controls import contiguous_unit_slices, history_windows  # noqa: E402


TRAIN_UNITS = ("1_3", "1_4", "2_1", "2_4", "2_5", "3_1", "3_2", "3_3")
VAL_UNITS = ("1_1", "1_2", "3_5")
TEST_UNITS = ("1_5", "2_2", "2_3", "3_4")
WINDOW = 50
TRAIN_STRIDE = 50
HORIZON_NAMES = ("5pct", "10pct", "20pct")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    """Encode undefined numeric metrics as JSON null, never non-standard NaN."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _unit_name(value: str) -> str:
    normalized = value.lower().replace("bearing", "").replace(" ", "").replace("-", "_")
    if normalized not in {*TRAIN_UNITS, *VAL_UNITS, *TEST_UNITS}:
        raise ValueError(f"Unexpected bearing name in PHMD split: {value!r}")
    return normalized


def _as_unit_list(value: Any, expected_count: int) -> list[np.ndarray] | None:
    import awkward as ak

    if isinstance(value, (list, tuple)):
        if len(value) == expected_count:
            return [
                np.asarray(ak.to_numpy(item) if isinstance(item, ak.Array) else item)
                for item in value
            ]
        return None
    if isinstance(value, ak.Array):
        as_list = ak.to_list(value)
        if len(as_list) == expected_count:
            return [np.asarray(item) for item in as_list]
        return None
    array = np.asarray(value)
    if array.ndim >= 3 and array.shape[0] == expected_count:
        return [array[index] for index in range(len(array))]
    return None


def _load_processed_units(
    repo_root: Path, config_path: Path
) -> tuple[dict[str, dict[str, tuple[np.ndarray, np.ndarray]]], dict[str, float]]:
    """Run the existing cache-aware PICID preprocessing protocol and return arrays."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from picid.data.data_objects import SplitViewPolicy
    from picid.data.preprocessing.preprocessor import PreProcessor
    from picid.transforms.base.transform_manager import ConfigTransformManager

    cfg = OmegaConf.load(config_path)
    source = instantiate(cfg.datasource)
    transforms = ConfigTransformManager(transforms_config=cfg.transforms)
    preprocessor = PreProcessor(datasource=source, transforms=transforms)
    cache_path = Path(str(cfg.paths.cache_path)).expanduser()
    root_path = Path(str(cfg.paths.root_dir)).expanduser()
    preprocessor.pipeline(
        data_cache_path=cache_path,
        data_library_part_path=root_path / "picid/data/datasources",
        transform_library_part_path=root_path / "picid/transforms",
        cache_preprocessed=bool(cfg.cache.use_cache_after_transfroms),
        use_boundary_cache=bool(
            cfg.cache.get("use_cache_after_boundary_conditions", True)
        ),
    )
    split_data = preprocessor.get_processed_split_dict(
        view_policy=SplitViewPolicy.KEEP_UNIT_LISTS
    )
    metadata = preprocessor.get_meta_data_dict()
    names_by_split: dict[str, list[str]] = {}
    components: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for split in ("train", "val", "test"):
        names = [_unit_name(str(value)) for value in metadata["unit_names"][split]]
        raw_values = {
            key: _as_unit_list(split_data[split][key], len(names))
            for key in (
                "time_domain_features",
                "spectral_features",
                "cumsum_feature",
                "target",
            )
        }
        if any(value is None for value in raw_values.values()):
            raise ValueError(f"Could not recover raw per-bearing features for {split}")
        components[split] = {}
        for index, name in enumerate(names):
            time_x = np.asarray(raw_values["time_domain_features"][index]).reshape(
                -1, 428
            )
            spectral_x = np.asarray(raw_values["spectral_features"][index]).reshape(
                -1, 30
            )
            cumsum_x = np.asarray(raw_values["cumsum_feature"][index]).reshape(-1, 2)
            raw_x = np.concatenate([spectral_x, time_x, cumsum_x], axis=1)
            raw_rul = np.asarray(raw_values["target"][index], dtype=np.float64).reshape(
                -1
            )
            if len(raw_x) != len(raw_rul):
                raise ValueError(f"Raw target/feature rows are misaligned for {name}")
            if not np.array_equal(raw_rul, np.arange(len(raw_rul) - 1, -1, -1)):
                raise ValueError(f"PHMD RUL timeline mismatch for {name}")
            components[split][name] = {"features": raw_x, "raw_rul": raw_rul}
        names_by_split[split] = names
        expected = {"train": TRAIN_UNITS, "val": VAL_UNITS, "test": TEST_UNITS}[split]
        if set(names) != set(expected):
            raise ValueError(f"{split} PHMD membership mismatch: {sorted(names)}")

    train_rows = np.concatenate(
        [components["train"][name]["features"] for name in names_by_split["train"]]
    )
    final_scaler = MinMaxScaler().fit(train_rows)
    from picid.transforms.bearings.health_index import DEFAULT_TOTAL_LIFE_LOOKUP

    output: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    cache_input_max_error: dict[str, float] = {}
    for split in ("train", "val", "test"):
        cached_features = np.asarray(split_data[split]["features"]).squeeze()
        cached_target = np.asarray(split_data[split]["rul"]).squeeze()
        cached_unit_ids = np.asarray(split_data[split]["unit_id"]).squeeze()
        if cached_features.ndim == 1:
            cached_features = cached_features[None, :]
        if cached_target.ndim == 1:
            cached_target = cached_target[:, None]
        groups = contiguous_unit_slices(cached_unit_ids)
        names = names_by_split[split]
        if len(groups) != len(names):
            raise ValueError(
                f"{split} row-aligned unit IDs do not match metadata device count"
            )
        units: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        max_error = 0.0
        for (_, row_slice), name in zip(groups, names, strict=True):
            item = components[split][name]
            scaled = final_scaler.transform(item["features"]).astype(np.float32)
            full_x = history_windows(scaled, np.arange(len(scaled)), WINDOW).reshape(
                len(scaled), -1
            )
            life = float(
                DEFAULT_TOTAL_LIFE_LOOKUP["XJTU-SY"][tuple(map(int, name.split("_")))]
            )
            full_y = (item["raw_rul"] / life).astype(np.float32)
            selected = (
                np.arange(0, len(full_x), TRAIN_STRIDE, dtype=np.int64)
                if split == "train"
                else np.arange(len(full_x), dtype=np.int64)
            )
            regenerated_x, regenerated_y = full_x[selected], full_y[selected]
            saved_x = cached_features[row_slice].reshape(
                row_slice.stop - row_slice.start, -1
            )
            saved_y = cached_target[row_slice].reshape(-1)
            if len(saved_x) != len(regenerated_x) or len(saved_y) != len(regenerated_y):
                raise ValueError(
                    f"Regenerated query count differs from cache for {name}"
                )
            max_error = max(max_error, float(np.max(np.abs(saved_x - regenerated_x))))
            if not np.allclose(saved_y, regenerated_y, atol=1e-6, rtol=0):
                raise ValueError(
                    f"Regenerated normalized HI target differs from cache for {name}"
                )
            units[name] = (full_x, full_y)
        if max_error > 1e-4:
            raise ValueError(
                f"Regenerated {split} windows differ from cache by {max_error}"
            )
        cache_input_max_error[split] = max_error
        output[split] = units
    del preprocessor
    return output, cache_input_max_error


def _training_samples(
    units: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    for name in sorted(units):
        x, y = units[name]
        # Build PICID's causal seq_len=50 histories from per-acquisition data,
        # then subsample endpoints to reproduce the selected train stride.
        ends = np.arange(0, len(x), TRAIN_STRIDE, dtype=np.int64)
        features.append(x[ends])
        targets.append(y[ends])
        ids.append(np.full(len(ends), name, dtype=object))
    return np.concatenate(features), np.concatenate(targets), np.concatenate(ids)


def _query_samples(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ends = np.arange(len(x), dtype=np.int64)
    return x, y.copy(), ends, np.arange(len(x) - 1, -1, -1, dtype=np.int64)


def _device_weights(unit_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(unit_ids).reshape(-1)
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights * (len(counts) / weights.sum())


def _last_step_features(
    units: dict[str, tuple[np.ndarray, np.ndarray]], name: str
) -> np.ndarray:
    x = units[name][0]
    if x.shape[1] % WINDOW:
        raise ValueError(
            f"Processed feature width {x.shape[1]} is not divisible by window {WINDOW}"
        )
    return x.reshape(len(x), WINDOW, -1)[:, -1, :]


def _logits_borders(full_prediction: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(full_prediction, dict):
        logits, borders = full_prediction["logits"], full_prediction["borders"]
    else:
        logits, borders = full_prediction.logits, full_prediction.borders
    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    if hasattr(borders, "detach"):
        borders = borders.detach().cpu().numpy()
    return np.asarray(logits), np.asarray(borders)


def _event_labels(
    units: dict[str, tuple[np.ndarray, np.ndarray]], horizons: dict[str, float]
) -> dict[str, np.ndarray]:
    return {
        name: warning_labels_from_rul(
            np.arange(len(x) - 1, -1, -1, dtype=np.float64), horizons
        )
        for name, (x, _) in units.items()
    }


def _stack_by_device(
    per_device: dict[str, np.ndarray],
    labels: dict[str, dict[str, np.ndarray]],
    horizon: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = sorted(per_device)
    return (
        np.concatenate([per_device[name] for name in names]),
        np.concatenate([labels[name][horizon].astype(np.int64) for name in names]),
        np.concatenate(
            [np.full(len(per_device[name]), name, dtype=object) for name in names]
        ),
    )


def _crossfit_platt(
    scores: np.ndarray,
    labels: np.ndarray,
    unit_ids: np.ndarray,
    *,
    score_is_probability: bool,
) -> np.ndarray:
    result = np.empty(len(scores), dtype=np.float64)
    for held_out in np.unique(unit_ids):
        train_mask = unit_ids != held_out
        test_mask = ~train_mask
        model = fit_platt_calibrator(
            scores[train_mask],
            labels[train_mask],
            unit_ids[train_mask],
            score_is_probability=score_is_probability,
        )
        result[test_mask] = apply_platt_calibrator(
            model,
            scores[test_mask],
            score_is_probability=score_is_probability,
        )
    return result


def _direct_event_oof(
    units: dict[str, tuple[np.ndarray, np.ndarray]],
    labels: dict[str, dict[str, np.ndarray]],
    horizon: str,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    all_names = sorted(units)
    for held_out in all_names:
        train_x: list[np.ndarray] = []
        train_y: list[np.ndarray] = []
        train_ids: list[np.ndarray] = []
        for name in all_names:
            if name == held_out:
                continue
            train_x.append(_last_step_features(units, name))
            train_y.append(labels[name][horizon].astype(np.int64))
            train_ids.append(np.full(len(labels[name][horizon]), name, dtype=object))
        x_train = np.concatenate(train_x)
        y_train = np.concatenate(train_y)
        ids_train = np.concatenate(train_ids)
        if np.unique(y_train).size != 2:
            raise ValueError(f"Direct event fold for {horizon} lacks both classes")
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        model.fit(x_train, y_train, sample_weight=_device_weights(ids_train))
        output[held_out] = model.predict_proba(_last_step_features(units, held_out))[
            :, 1
        ]
    return output


def _direct_event_test(
    train_units: dict[str, tuple[np.ndarray, np.ndarray]],
    train_labels: dict[str, dict[str, np.ndarray]],
    test_units: dict[str, tuple[np.ndarray, np.ndarray]],
    horizon: str,
) -> dict[str, np.ndarray]:
    x_train = np.concatenate(
        [_last_step_features(train_units, name) for name in sorted(train_units)]
    )
    y_train = np.concatenate(
        [train_labels[name][horizon] for name in sorted(train_units)]
    ).astype(np.int64)
    ids_train = np.concatenate(
        [
            np.full(len(train_units[name][0]), name, dtype=object)
            for name in sorted(train_units)
        ]
    )
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(x_train, y_train, sample_weight=_device_weights(ids_train))
    return {
        name: model.predict_proba(_last_step_features(test_units, name))[:, 1]
        for name in test_units
    }


def _bootstrap_device_means(
    per_device: dict[str, float], *, seed: int, repetitions: int = 10000
) -> dict[str, float | int]:
    values = np.asarray(list(per_device.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[sampled].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "bearing_count": int(len(values)),
        "repetitions": repetitions,
        "percentile_95_low": float(low),
        "percentile_95_high": float(high),
        "scope": "four held-out bearings; interval describes device resampling only",
    }


def _metric_device_values(
    probabilities: np.ndarray,
    labels: np.ndarray,
    unit_ids: np.ndarray,
    metric: str,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for unit in np.unique(unit_ids):
        mask = unit_ids == unit
        p, y = probabilities[mask], labels[mask]
        eps = np.finfo(np.float64).eps
        if metric == "brier":
            value = np.mean((p - y) ** 2)
        else:
            p = np.clip(p, eps, 1 - eps)
            value = np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
        result[str(unit)] = float(value)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--horizons", type=Path, required=True)
    parser.add_argument("--weight-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--n-ensembles", type=int, default=8)
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(args.physical_gpu) or args.physical_gpu not in range(1, 7):
        raise RuntimeError(
            "Set CUDA_VISIBLE_DEVICES to exactly one authorized physical GPU "
            f"1-6; received requested={args.physical_gpu}, visible={visible!r}"
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

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "status": "running",
        "command": shlex.join(sys.argv),
        "cwd": os.getcwd(),
        "source_root": str(repo_root),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "source_sha256": {
            "script": _sha256(Path(__file__).resolve()),
            "prediction_io": _sha256(repo_root / "picid/research/prediction_io.py"),
            "warning_metrics": _sha256(repo_root / "picid/research/warning_metrics.py"),
            "warning_probabilities": _sha256(
                repo_root / "picid/research/warning_probabilities.py"
            ),
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
        "tabdpt_version": version("tabdpt"),
        "weight_path": str(args.weight_path.resolve()),
        "weight_sha256": _sha256(args.weight_path),
        "base_model_cache": str(args.model_cache.resolve()),
        "base_model_cache_sha256": _sha256(args.model_cache),
        "baseline_prediction_path": str(args.baseline_predictions.resolve()),
        "baseline_prediction_sha256": _sha256(args.baseline_predictions),
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "horizon_path": str(args.horizons.resolve()),
        "horizon_sha256": _sha256(args.horizons),
        "parameters": {
            "window": WINDOW,
            "training_stride": TRAIN_STRIDE,
            "n_ensembles": args.n_ensembles,
            "context_size": args.context_size,
            "batch_size": args.batch_size,
            "prediction_interface": "TabDPT 1.3.0 output_type=full",
            "full_output_ensemble_reduction": "mean member logits, then softmax for CDF",
            "target_fpr": args.target_fpr,
            "calibration": "leave-one-training-bearing-out TabDPT predictions; leave-one-bearing-out Platt scores for threshold fitting",
            "direct_model": "device-weighted logistic regression C=1 on current processed features",
        },
        "test_data_boundary": "test RUL/time used only for retrospective warning labels, event episodes, and lead time; model scores use only test features and the training-fitted model",
        "test_history_disclosure": "public PHMD test trajectories were accessed in prior baseline model runs; these warning metrics are not a blind external validation",
        "undefined_metric_encoding": "JSON null denotes undefined metrics such as a device with no negative-warning query for false-alarm FPR",
    }
    _atomic_json(manifest_path, manifest)
    if free_bytes < expected_peak_gib * 1024**3:
        manifest["status"] = "failed_resource_preflight"
        _atomic_json(manifest_path, manifest)
        raise RuntimeError(
            f"Only {free_bytes / 1024**3:.2f} GiB free; need conservative "
            f"{expected_peak_gib:.1f} GiB for this model configuration"
        )
    started = time.perf_counter()

    horizon_payload = json.loads(args.horizons.read_text())
    horizons = {
        key: float(horizon_payload["horizons"][key]["rul_threshold"])
        for key in HORIZON_NAMES
    }
    median_lifetime = float(horizon_payload["median_training_device_max_rul"])
    if median_lifetime != 354.0:
        raise ValueError(
            f"The frozen XJTU training median should be 354, got {median_lifetime}"
        )
    data, cache_input_alignment = _load_processed_units(repo_root, args.config)
    train_units, val_units, test_units = data["train"], data["val"], data["test"]
    manifest["regenerated_input_vs_frozen_cache_max_abs_error"] = cache_input_alignment
    if len(val_units) != len(VAL_UNITS):
        raise ValueError(
            f"Expected {len(VAL_UNITS)} validation bearings, got {len(val_units)}"
        )
    manifest["processed_unit_lengths"] = {
        split: {name: int(len(x)) for name, (x, _) in units.items()}
        for split, units in data.items()
    }
    manifest["processed_feature_dimensions"] = {
        split: {name: int(x.shape[1]) for name, (x, _) in units.items()}
        for split, units in data.items()
    }
    manifest["processed_input_sha256"] = {
        split: {
            name: {
                "features": hashlib.sha256(
                    np.ascontiguousarray(x).view(np.uint8)
                ).hexdigest(),
                "normalized_hi": hashlib.sha256(
                    np.ascontiguousarray(y).view(np.uint8)
                ).hexdigest(),
            }
            for name, (x, y) in units.items()
        }
        for split, units in data.items()
    }
    _atomic_json(manifest_path, manifest)

    if args.preflight_only:
        held_out = max(train_units, key=lambda name: len(train_units[name][0]))
        fit_units = {
            name: value for name, value in train_units.items() if name != held_out
        }
        fit_x, fit_y, _ = _training_samples(fit_units)
        query_x = train_units[held_out][0]
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
        model.fit(fit_x, fit_y)
        torch.cuda.synchronize()
        fit_seconds = time.perf_counter() - fit_started
        predict_started = time.perf_counter()
        oof_pilot = model.predict(
            query_x,
            output_type="full",
            seed=args.seed,
            context_size=args.context_size,
            n_ensembles=args.n_ensembles,
            batch_size=args.batch_size,
        )
        torch.cuda.synchronize()
        oof_predict_seconds = time.perf_counter() - predict_started
        del model, oof_pilot, fit_x, fit_y
        gc.collect()
        torch.cuda.empty_cache()

        import cloudpickle

        with args.model_cache.open("rb") as stream:
            cache_model = cloudpickle.load(stream)
        largest_test = max(test_units, key=lambda name: len(test_units[name][0]))
        torch.cuda.reset_peak_memory_stats(0)
        test_started = time.perf_counter()
        test_pilot = cache_model.predict(
            test_units[largest_test][0],
            output_type="full",
            seed=args.seed,
            context_size=args.context_size,
            n_ensembles=args.n_ensembles,
            batch_size=args.batch_size,
        )
        torch.cuda.synchronize()
        test_predict_seconds = time.perf_counter() - test_started
        logits, borders = _logits_borders(test_pilot)
        pilot_mean = tabdpt_distribution_mean(logits, borders)
        peak_allocated_gib = float(torch.cuda.max_memory_allocated(0) / 1024**3)
        del cache_model, test_pilot, logits, borders
        gc.collect()
        torch.cuda.empty_cache()
        manifest["status"] = "preflight_complete"
        manifest["preflight"] = {
            "largest_oof_bearing": held_out,
            "oof_fit_seconds": fit_seconds,
            "oof_predict_seconds": oof_predict_seconds,
            "oof_query_count": int(len(query_x)),
            "largest_test_bearing": largest_test,
            "test_predict_seconds": test_predict_seconds,
            "test_query_count": int(len(test_units[largest_test][0])),
            "peak_allocated_gib": peak_allocated_gib,
            "prediction_finite": bool(np.isfinite(pilot_mean).all()),
            "conservative_full_study_runtime_estimate_seconds": float(
                len(train_units) * (fit_seconds + oof_predict_seconds)
                + sum(len(test_units[name][0]) for name in test_units)
                / len(test_units[largest_test][0])
                * test_predict_seconds
            ),
        }
        manifest["wall_seconds"] = time.perf_counter() - started
        manifest["gpu_snapshot_after_finish"] = _snapshot_gpu_state()
        _atomic_json(manifest_path, manifest)
        return 0

    train_labels = _event_labels(train_units, horizons)
    test_labels = _event_labels(test_units, horizons)
    oof_native: dict[str, dict[str, np.ndarray]] = {h: {} for h in HORIZON_NAMES}
    oof_mean: dict[str, np.ndarray] = {}
    oof_direct: dict[str, dict[str, np.ndarray]] = {h: {} for h in HORIZON_NAMES}
    fold_summaries = []
    prediction_archive_arrays: dict[str, np.ndarray] = {}
    oof_cache_root = output_dir / "oof_fold_cache"
    oof_cache_root.mkdir(parents=True, exist_ok=True)
    run_identity = hashlib.sha256(
        json.dumps(
            {
                "config": _sha256(args.config),
                "horizons": _sha256(args.horizons),
                "weight": _sha256(args.weight_path),
                "script": _sha256(Path(__file__).resolve()),
                "processed_inputs": manifest["processed_input_sha256"],
                "seed": args.seed,
                "window": WINDOW,
                "stride": TRAIN_STRIDE,
                "n_ensembles": args.n_ensembles,
                "context_size": args.context_size,
                "batch_size": args.batch_size,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    fold_started = time.perf_counter()
    for held_out in sorted(train_units):
        fold_cache_path = oof_cache_root / f"{held_out}_oof.npz"
        if fold_cache_path.exists():
            with np.load(fold_cache_path) as saved:
                if str(saved["run_identity"].item()) == run_identity:
                    oof_mean[held_out] = np.asarray(saved["mean"], dtype=np.float64)
                    for horizon_name in HORIZON_NAMES:
                        oof_native[horizon_name][held_out] = np.asarray(
                            saved[horizon_name], dtype=np.float64
                        )
                    fold_summaries.append(json.loads(str(saved["summary"].item())))
                    print(f"resumed cached OOF fold {held_out}", flush=True)
                    continue
        fit_units = {
            name: value for name, value in train_units.items() if name != held_out
        }
        fit_x, fit_y, fit_ids = _training_samples(fit_units)
        query_x, query_y, query_times, rul = _query_samples(*train_units[held_out])
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
        fit_start = time.perf_counter()
        model.fit(fit_x, fit_y)
        torch.cuda.synchronize()
        fit_seconds = time.perf_counter() - fit_start
        pred_start = time.perf_counter()
        full = model.predict(
            query_x,
            output_type="full",
            seed=args.seed,
            context_size=args.context_size,
            n_ensembles=args.n_ensembles,
            batch_size=args.batch_size,
        )
        torch.cuda.synchronize()
        predict_seconds = time.perf_counter() - pred_start
        logits, borders = _logits_borders(full)
        mean = tabdpt_distribution_mean(logits, borders)
        expected_queries = len(train_units[held_out][0])
        if len(mean) != expected_queries or not np.isfinite(mean).all():
            raise ValueError(
                f"OOF distribution does not align with held-out device {held_out}"
            )
        oof_mean[held_out] = mean
        native_by_horizon = {}
        for horizon_name in HORIZON_NAMES:
            proxy_threshold = horizons[horizon_name] / median_lifetime
            native_by_horizon[horizon_name] = tabdpt_distribution_cdf(
                logits, borders, proxy_threshold
            )
            oof_native[horizon_name][held_out] = native_by_horizon[horizon_name]
        training_context_rows = len(fit_y)
        fold_summary = {
            "held_out_device": held_out,
            "training_device_count": len(fit_units),
            "training_context_rows": int(training_context_rows),
            "query_count": int(expected_queries),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "peak_allocated_gib": float(torch.cuda.max_memory_allocated(0) / 1024**3),
        }
        fold_payload = {
            "run_identity": np.asarray(run_identity),
            "mean": mean,
            "summary": np.asarray(json.dumps(fold_summary)),
            **native_by_horizon,
        }
        temporary_fold = fold_cache_path.with_suffix(".npz.tmp")
        with temporary_fold.open("wb") as stream:
            np.savez_compressed(stream, **fold_payload)
        temporary_fold.replace(fold_cache_path)
        del model, full, logits, borders, fit_x, fit_y, fit_ids, query_x
        gc.collect()
        torch.cuda.empty_cache()
        fold_summaries.append(fold_summary)
        manifest["oof_completed_folds"] = fold_summaries
        _atomic_json(manifest_path, manifest)
        print(f"completed leave-one-bearing-out fold {held_out}", flush=True)
    manifest["oof_wall_seconds"] = time.perf_counter() - fold_started
    manifest["oof_folds"] = fold_summaries

    for horizon_name in HORIZON_NAMES:
        oof_direct[horizon_name] = _direct_event_oof(
            train_units, train_labels, horizon_name
        )

    baseline_ids, baseline_targets, baseline_preds = load_normalized_predictions(
        args.baseline_predictions
    )
    unit_id_map = {"1_5": (1, 5), "2_2": (2, 2), "2_3": (2, 3), "3_4": (3, 4)}
    expected_test_ids = np.asarray([unit_id_map[name] for name in sorted(TEST_UNITS)])
    if not {tuple(row) for row in np.unique(baseline_ids, axis=0)} == {
        tuple(row) for row in expected_test_ids
    }:
        raise ValueError(
            "Saved baseline predictions do not match the frozen PHMD test bearings"
        )

    import cloudpickle

    with args.model_cache.open("rb") as stream:
        cache_model = cloudpickle.load(stream)
    test_native: dict[str, dict[str, np.ndarray]] = {h: {} for h in HORIZON_NAMES}
    test_mean: dict[str, np.ndarray] = {}
    test_prediction_seconds = 0.0
    test_peak_allocated_gib = 0.0
    for name in sorted(test_units):
        x, y = test_units[name]
        query_x, query_y, times, rul = _query_samples(x, y)
        torch.cuda.reset_peak_memory_stats(0)
        prediction_started = time.perf_counter()
        full = cache_model.predict(
            query_x,
            output_type="full",
            seed=args.seed,
            context_size=args.context_size,
            n_ensembles=args.n_ensembles,
            batch_size=args.batch_size,
        )
        torch.cuda.synchronize()
        test_prediction_seconds += time.perf_counter() - prediction_started
        test_peak_allocated_gib = max(
            test_peak_allocated_gib,
            float(torch.cuda.max_memory_allocated(0) / 1024**3),
        )
        logits, borders = _logits_borders(full)
        mean = tabdpt_distribution_mean(logits, borders)
        test_mean[name] = mean
        for horizon_name in HORIZON_NAMES:
            test_native[horizon_name][name] = tabdpt_distribution_cdf(
                logits, borders, horizons[horizon_name] / median_lifetime
            )
        del full, logits, borders, query_x
        gc.collect()
    del cache_model
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Check point-distribution identity against the already recorded evaluation.
    baseline_max_error = 0.0
    baseline_max_error_by_device: dict[str, float] = {}
    baseline_targets_max_error = 0.0
    for name in sorted(test_units):
        expected_ids = unit_id_map[name]
        mask = np.all(baseline_ids == np.asarray(expected_ids), axis=1)
        expected = baseline_preds[mask]
        if len(expected) != len(test_mean[name]):
            raise ValueError(
                f"Test rows do not align for {name}: {len(expected)} vs {len(test_mean[name])}"
            )
        device_max_error = float(np.max(np.abs(expected - test_mean[name])))
        baseline_max_error_by_device[name] = device_max_error
        baseline_max_error = max(baseline_max_error, device_max_error)
        target_error = np.max(np.abs(baseline_targets[mask] - test_units[name][1]))
        baseline_targets_max_error = max(
            baseline_targets_max_error, float(target_error)
        )
        if target_error > 1e-6:
            raise ValueError(
                f"Processed normalized HI target does not align with saved result for {name}"
            )
    manifest["baseline_distribution_mean_max_abs_error"] = baseline_max_error
    point_predictions_equivalent = all(
        np.allclose(
            baseline_preds[
                np.all(baseline_ids == np.asarray(unit_id_map[name]), axis=1)
            ],
            test_mean[name],
            atol=1e-4,
            rtol=1e-4,
        )
        for name in sorted(test_units)
    )
    manifest["point_prediction_equivalence"] = {
        "status": (
            "within_tolerance"
            if point_predictions_equivalent
            else "separate_full_output_protocol"
        ),
        "atol": 1e-4,
        "rtol": 1e-4,
        "merge_with_saved_point_baseline": point_predictions_equivalent,
        "per_device_max_abs_error": baseline_max_error_by_device,
        "overall_max_abs_error": baseline_max_error,
        "interpretation": (
            "Full-distribution and saved point-prediction outputs match within tolerance."
            if point_predictions_equivalent
            else "TabDPT full output averages ensemble logits before probability conversion, while the saved point path averages member-wise expectations; retain as a separate inference protocol."
        ),
    }
    manifest["baseline_target_max_abs_error"] = baseline_targets_max_error
    manifest["test_distribution_predict_seconds"] = test_prediction_seconds
    manifest["test_distribution_peak_allocated_gib"] = test_peak_allocated_gib

    for horizon_name in HORIZON_NAMES:
        # Device-wise OOF event labels and raw scores.
        native_cal_x, native_cal_y, native_cal_ids = _stack_by_device(
            oof_native[horizon_name], train_labels, horizon_name
        )
        residual_groups = {
            name: train_units[name][1] - oof_mean[name] for name in sorted(train_units)
        }
        residual_oof_by_device = {
            name: residual_ecdf_score(
                residual_groups,
                oof_mean[name],
                horizons[horizon_name] / median_lifetime,
                exclude_device=name,
            )
            for name in sorted(train_units)
        }
        residual_cal_x, residual_cal_y, residual_cal_ids = _stack_by_device(
            residual_oof_by_device, train_labels, horizon_name
        )
        direct_cal_x, direct_cal_y, direct_cal_ids = _stack_by_device(
            oof_direct[horizon_name], train_labels, horizon_name
        )
        for label_name, label_values in (
            ("native", native_cal_y),
            ("residual", residual_cal_y),
            ("direct", direct_cal_y),
        ):
            if not np.array_equal(label_values, native_cal_y):
                raise AssertionError(f"OOF calibration labels differ for {label_name}")

        oof_probabilities = {
            "native_cdf": _crossfit_platt(
                native_cal_x, native_cal_y, native_cal_ids, score_is_probability=True
            ),
            "residual_ecdf": _crossfit_platt(
                residual_cal_x,
                residual_cal_y,
                residual_cal_ids,
                score_is_probability=True,
            ),
            "direct_logistic": _crossfit_platt(
                direct_cal_x, direct_cal_y, direct_cal_ids, score_is_probability=True
            ),
        }
        test_raw_by_method: dict[str, dict[str, np.ndarray]] = {
            "native_cdf": test_native[horizon_name],
            "residual_ecdf": {
                name: residual_ecdf_score(
                    residual_groups,
                    test_mean[name],
                    horizons[horizon_name] / median_lifetime,
                )
                for name in sorted(test_units)
            },
            "direct_logistic": _direct_event_test(
                train_units, train_labels, test_units, horizon_name
            ),
        }
        test_label_arr = np.concatenate(
            [
                test_labels[name][horizon_name].astype(np.int64)
                for name in sorted(test_units)
            ]
        )
        test_ids_arr = np.concatenate(
            [
                np.full(len(test_units[name][0]), name, dtype=object)
                for name in sorted(test_units)
            ]
        )
        test_rul_arr = np.concatenate(
            [
                np.arange(len(test_units[name][0]) - 1, -1, -1, dtype=np.float64)
                for name in sorted(test_units)
            ]
        )
        test_time_arr = np.concatenate(
            [
                np.arange(len(test_units[name][0]), dtype=np.int64)
                for name in sorted(test_units)
            ]
        )

        methods: dict[str, Any] = {}
        for method, (cal_x, cal_y, cal_ids) in {
            "native_cdf": (native_cal_x, native_cal_y, native_cal_ids),
            "residual_ecdf": (residual_cal_x, residual_cal_y, residual_cal_ids),
            "direct_logistic": (direct_cal_x, direct_cal_y, direct_cal_ids),
        }.items():
            calibrator = fit_platt_calibrator(
                cal_x, cal_y, cal_ids, score_is_probability=True
            )
            test_raw = np.concatenate(
                [test_raw_by_method[method][name] for name in sorted(test_units)]
            )
            test_probability = apply_platt_calibrator(
                calibrator, test_raw, score_is_probability=True
            )
            oof_probability = oof_probabilities[method]
            threshold, calibration_fpr = select_empirical_fpr_threshold(
                oof_probability,
                native_cal_y,
                native_cal_ids,
                target_fpr=args.target_fpr,
            )
            probability_summary = probability_metrics(
                test_probability, test_label_arr, test_ids_arr
            )
            alarm = alarm_metrics(
                test_probability,
                test_label_arr,
                test_rul_arr,
                test_ids_arr,
                test_time_arr,
                threshold=threshold,
            )
            device_brier = _metric_device_values(
                test_probability, test_label_arr, test_ids_arr, "brier"
            )
            device_logloss = _metric_device_values(
                test_probability, test_label_arr, test_ids_arr, "logloss"
            )
            per_device_detection = {
                name: float(alarm.per_device[name]["detected_in_warning_window"])
                for name in alarm.per_device
            }
            per_device_false_alarms = {
                name: float(alarm.per_device[name]["false_alarm_episode_count"])
                for name in alarm.per_device
            }
            methods[method] = {
                "raw_test_probability_summary": probability_metrics(
                    np.clip(test_raw, 0.0, 1.0), test_label_arr, test_ids_arr
                ),
                "calibrated_test_probability_metrics": probability_summary,
                "oof_threshold": threshold,
                "oof_empirical_device_macro_fpr": calibration_fpr,
                "test_alarm_metrics": alarm.metrics,
                "test_per_device_alarm_metrics": alarm.per_device,
                "device_bootstrap_95pct": {
                    "device_macro_brier": _bootstrap_device_means(
                        device_brier, seed=args.seed
                    ),
                    "device_macro_log_loss": _bootstrap_device_means(
                        device_logloss, seed=args.seed + 1
                    ),
                    "device_detection_rate": _bootstrap_device_means(
                        per_device_detection, seed=args.seed + 2
                    ),
                    "false_alarm_episodes_per_device": _bootstrap_device_means(
                        per_device_false_alarms, seed=args.seed + 3
                    ),
                },
            }
            prediction_archive_arrays[f"{horizon_name}_{method}_raw"] = test_raw
            prediction_archive_arrays[f"{horizon_name}_{method}_calibrated"] = (
                test_probability
            )
        manifest.setdefault("horizon_results", {})[horizon_name] = {
            "raw_warning_horizon_intervals": horizons[horizon_name],
            "training_median_lifetime_intervals": median_lifetime,
            "distribution_cdf_proxy_threshold": horizons[horizon_name]
            / median_lifetime,
            "calibration_device_count": len(train_units),
            "evaluation_device_count": len(test_units),
            "evaluation": methods,
        }
        _atomic_json(manifest_path, manifest)

    archive = output_dir / f"xjtu_warning_seed{args.seed}_predictions.npz"
    archive_arrays = prediction_archive_arrays
    for name in sorted(test_units):
        archive_arrays[f"{name}_normalized_hi_mean"] = test_mean[name]
        for horizon in HORIZON_NAMES:
            archive_arrays[f"{name}_{horizon}_native_cdf"] = test_native[horizon][name]
    np.savez_compressed(archive, **archive_arrays)
    manifest["prediction_archive"] = str(archive.resolve())
    manifest["prediction_archive_sha256"] = _sha256(archive)
    manifest["status"] = "complete"
    manifest["wall_seconds"] = time.perf_counter() - started
    manifest["gpu_snapshot_after_finish"] = _snapshot_gpu_state()
    _atomic_json(manifest_path, manifest)
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
                    _atomic_json(target, payload)
                except Exception:
                    pass
        raise
    raise SystemExit(exit_code)
