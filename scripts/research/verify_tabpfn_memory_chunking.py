"""Check prediction equivalence when only TabPFN's memory chunk factor changes."""
import argparse
import gc
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from tabpfn.architectures.base.memory import MemoryUsageEstimator
from tabpfn.regressor import TabPFNRegressor


def predict_at_factor(factor, x_train, y_train, x_query):
    MemoryUsageEstimator.SAVE_PEAK_MEM_FACTOR = factor
    torch.cuda.reset_peak_memory_stats()
    estimator = TabPFNRegressor(
        device="cuda",
        n_estimators=1,
        random_state=72,
        n_jobs=1,
        fit_mode="fit_with_cache_single_step_inference",
        memory_saving_mode=True,
        ignore_pretraining_limits=True,
    )
    estimator.fit(x_train, y_train)
    prediction = np.asarray(estimator.predict(x_query), dtype=np.float64).reshape(-1)
    peak_mib = torch.cuda.max_memory_allocated() // (1024 * 1024)
    del estimator
    gc.collect()
    torch.cuda.empty_cache()
    return prediction, int(peak_mib)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    rng = np.random.default_rng(72)
    x_train = rng.normal(size=(256, 360)).astype(np.float32)
    y_train = (np.sin(x_train[:, 0]) + 0.1 * x_train[:, 1]).astype(np.float32)
    x_query = rng.normal(size=(32, 360)).astype(np.float32)
    prediction8, peak8 = predict_at_factor(8, x_train, y_train, x_query)
    prediction16, peak16 = predict_at_factor(16, x_train, y_train, x_query)
    difference = np.abs(prediction8 - prediction16)
    report = {
        "tabpfn_version": version("tabpfn"),
        "seed": 72,
        "n_train": int(len(x_train)),
        "n_features": int(x_train.shape[1]),
        "n_query": int(len(x_query)),
        "n_estimators": 1,
        "fit_mode": "fit_with_cache_single_step_inference",
        "memory_saving_mode": True,
        "factor8_peak_allocated_mib": peak8,
        "factor16_peak_allocated_mib": peak16,
        "max_absolute_difference": float(difference.max()),
        "mean_absolute_difference": float(difference.mean()),
        "atol": args.atol,
        "rtol": args.rtol,
        "equivalent": bool(np.allclose(prediction8, prediction16, atol=args.atol, rtol=args.rtol)),
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["equivalent"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
