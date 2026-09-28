"""Measure fixed-input numerical replay, batching cost and peak GPU allocations."""
import argparse
from functools import partial
import gc
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from picid.research.protocol import PHYSICAL_GPUS  # noqa: E402


def _predict_batches(model, queries, batch_size):
    return np.concatenate([model.predict(queries[i:i + batch_size])
                          for i in range(0, len(queries), batch_size)])


def _predict_tabpfn(model, queries, batch_size):
    return _predict_batches(model, queries, batch_size)


def _predict_tabdpt(model, queries, batch_size, modern):
    options = {"n_ensembles": 8, "seed": 72,
               "context_size": 2048}
    if modern:
        options["batch_size"] = batch_size
    return model.predict(queries, **options)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("tabpfn", "tabdpt", "tabdpt130"), required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-mode", default="fit_preprocessors")
    parser.add_argument("--train-rows", type=int, default=2049)
    parser.add_argument("--query-rows", type=int, default=512)
    parser.add_argument("--batches", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.isdigit() or int(visible) not in PHYSICAL_GPUS:
        raise ValueError(f"Explicit single allowed physical GPU required: {PHYSICAL_GPUS}")
    torch.set_num_threads(8)
    rng = np.random.default_rng(72)
    if args.inputs:
        X = np.load(args.inputs / "train_features.npy", mmap_mode="r")[:args.train_rows].copy()
        y = np.load(args.inputs / "train_rul.npy", mmap_mode="r")[:args.train_rows].reshape(-1).copy()
        Q = np.load(args.inputs / "val_features.npy", mmap_mode="r")[:args.query_rows].copy()
    else:
        X = rng.normal(size=(args.train_rows, 14)).astype("float32")
        y = (X[:, 0] * .1 + X[:, 1] * .2).astype("float32")
        Q = rng.normal(size=(args.query_rows, 14)).astype("float32")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"model": args.model, "physical_gpu": int(visible), "logical_gpu": "cuda:0",
              "seed": 72, "training_shape": X.shape, "query_shape": Q.shape,
              "input_source": str(args.inputs) if args.inputs else "synthetic_smoke_not_formal",
              "n_ensembles": 8, "atol": 1e-4, "rtol": 1e-4, "fit_mode": args.fit_mode,
              "version": version("tabpfn" if args.model == "tabpfn" else "tabdpt"), "batches": []}
    baseline = None
    for batch in args.batches:
        model = None
        predict = None
        record = {"batch_size": batch, "status": "running"}
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            if args.model == "tabpfn":
                from tabpfn import TabPFNRegressor
                model = TabPFNRegressor(device="cuda", model_path=str(args.weights), n_estimators=8,
                    random_state=72, ignore_pretraining_limits=True, n_jobs=8, fit_mode=args.fit_mode)
                model.fit(X, y)
                predict = partial(_predict_tabpfn, model, Q, batch)
                record["actual_context_rows"] = len(X)
            else:
                from tabdpt import TabDPTRegressor
                options = dict(device="cuda", model_weight_path=str(args.weights), compile=False)
                if args.model == "tabdpt":
                    options["inf_batch_size"] = batch
                model = TabDPTRegressor(**options)
                model.fit(X, y)
                predict = partial(_predict_tabdpt, model, Q, batch, args.model == "tabdpt130")
                record["actual_context_rows"] = min(len(X), 2048) if args.model == "tabdpt" else len(X)
            torch.cuda.synchronize()
            record["fit_seconds"] = time.monotonic() - start
            start = time.monotonic()
            pred = np.asarray(predict()).reshape(-1)
            torch.cuda.synchronize()
            record["cold_predict_seconds"] = time.monotonic() - start
            start = time.monotonic()
            replay = np.asarray(predict()).reshape(-1)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            record.update(steady_predict_seconds=elapsed, rows_per_second=len(Q)/elapsed,
                          peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                          peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                          replay_equal=bool(np.allclose(pred,replay,atol=1e-4,rtol=1e-4)))
            if baseline is None:
                baseline = pred
            record["batch_equal"] = bool(np.allclose(baseline,pred,atol=1e-4,rtol=1e-4))
            record["max_abs_diff"] = float(np.max(np.abs(baseline-pred)))
            record["status"] = "passed" if record["batch_equal"] and record["replay_equal"] else "not_equivalent"
            np.save(args.output / f"predictions_b{batch}.npy", pred)
        except torch.OutOfMemoryError as exc:
            record.update(status="oom", error=str(exc))
        except Exception as exc:
            record.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        report["batches"].append(record)
        (args.output / "benchmark.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(record), flush=True)
    if any(r["status"] != "passed" for r in report["batches"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
