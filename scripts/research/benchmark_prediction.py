"""Measure fixed-input numerical replay, batching cost and peak GPU allocations."""
import argparse
from functools import partial
import gc
import hashlib
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
from picid.model.estimators.tabpfn.context_sampling import (  # noqa: E402
    unit_balanced_temporal_indices,
)


def _predict_batches(model, queries, batch_size):
    return np.concatenate([model.predict(queries[i:i + batch_size])
                          for i in range(0, len(queries), batch_size)])


def _predict_tabpfn(model, queries, batch_size):
    return _predict_batches(model, queries, batch_size)


def _tabdpt_context_size(model_name):
    return None if model_name == "tabdpt120" else 2048


def _predict_tabdpt(model, queries, batch_size, uses_predict_batch_size, context_size):
    options = {"n_ensembles": 8, "seed": 72}
    if context_size is not None:
        options["context_size"] = context_size
    if uses_predict_batch_size:
        options["batch_size"] = batch_size
    return model.predict(queries, **options)


def prepare_benchmark_inputs(
    train_features,
    train_targets,
    val_features,
    *,
    train_unit_ids=None,
    val_unit_ids=None,
    train_rows=2049,
    query_rows=512,
    max_fit_samples=None,
):
    """Select a fixed fit context and query subset for batching comparisons."""
    if train_rows <= 0 or query_rows <= 0:
        raise ValueError("train_rows and query_rows must be positive")
    train_targets = np.asarray(train_targets).reshape(-1)
    if len(train_features) != len(train_targets):
        raise ValueError("training features and targets have different row counts")
    if train_unit_ids is not None and len(train_features) != len(train_unit_ids):
        raise ValueError("training unit IDs are not row-aligned with features")
    if val_unit_ids is not None and len(val_features) != len(val_unit_ids):
        raise ValueError("validation unit IDs are not row-aligned with features")

    if max_fit_samples is None:
        fit_indices = np.arange(min(train_rows, len(train_features)), dtype=np.int64)
        fit_selection = "chronological_prefix"
    else:
        if train_unit_ids is None:
            raise ValueError("unit_id metadata is required when max_fit_samples is set")
        fit_indices = unit_balanced_temporal_indices(train_unit_ids, max_fit_samples)
        fit_selection = "unit_balanced_temporal"

    if val_unit_ids is not None and query_rows < len(val_features):
        query_indices = unit_balanced_temporal_indices(val_unit_ids, query_rows)
        query_selection = "unit_balanced_temporal"
    else:
        query_indices = np.arange(min(query_rows, len(val_features)), dtype=np.int64)
        query_selection = "chronological_prefix"

    X = np.asarray(train_features[fit_indices]).copy()
    y = train_targets[fit_indices].copy()
    Q = np.asarray(val_features[query_indices]).copy()
    fit_ids = (
        np.asarray(train_unit_ids).reshape(len(train_unit_ids), -1)[fit_indices]
        if train_unit_ids is not None else None
    )
    query_ids = (
        np.asarray(val_unit_ids).reshape(len(val_unit_ids), -1)[query_indices]
        if val_unit_ids is not None else None
    )
    metadata = {
        "fit_selection": fit_selection,
        "query_selection": query_selection,
        "fit_rows": len(fit_indices),
        "query_rows": len(query_indices),
        "fit_indices_sha256": hashlib.sha256(fit_indices.tobytes()).hexdigest(),
        "query_indices_sha256": hashlib.sha256(query_indices.tobytes()).hexdigest(),
        "fit_unit_count": (
            len({tuple(row.tolist()) for row in fit_ids}) if fit_ids is not None else None
        ),
        "query_unit_count": (
            len({tuple(row.tolist()) for row in query_ids}) if query_ids is not None else None
        ),
    }
    return X, y, Q, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("tabpfn", "tabdpt", "tabdpt120", "tabdpt130"), required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-mode", default="fit_preprocessors")
    parser.add_argument("--train-rows", type=int, default=2049)
    parser.add_argument("--query-rows", type=int, default=512)
    parser.add_argument("--max-fit-samples", type=int)
    parser.add_argument("--batches", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.isdigit() or int(visible) not in PHYSICAL_GPUS:
        raise ValueError(f"Explicit single allowed physical GPU required: {PHYSICAL_GPUS}")
    torch.set_num_threads(8)
    rng = np.random.default_rng(72)
    input_metadata = None
    if args.inputs:
        train_features = np.load(args.inputs / "train_features.npy", mmap_mode="r")
        train_targets = np.load(args.inputs / "train_rul.npy", mmap_mode="r")
        val_features = np.load(args.inputs / "val_features.npy", mmap_mode="r")
        train_id_path = args.inputs / "train_unit_id.npy"
        val_id_path = args.inputs / "val_unit_id.npy"
        train_unit_ids = (
            np.load(train_id_path, mmap_mode="r") if train_id_path.exists() else None
        )
        val_unit_ids = np.load(val_id_path, mmap_mode="r") if val_id_path.exists() else None
        X, y, Q, input_metadata = prepare_benchmark_inputs(
            train_features,
            train_targets,
            val_features,
            train_unit_ids=train_unit_ids,
            val_unit_ids=val_unit_ids,
            train_rows=args.train_rows,
            query_rows=args.query_rows,
            max_fit_samples=args.max_fit_samples,
        )
    else:
        if args.max_fit_samples is not None:
            raise ValueError("--max-fit-samples requires --inputs with unit IDs")
        X = rng.normal(size=(args.train_rows, 14)).astype("float32")
        y = (X[:, 0] * .1 + X[:, 1] * .2).astype("float32")
        Q = rng.normal(size=(args.query_rows, 14)).astype("float32")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"model": args.model, "physical_gpu": int(visible), "logical_gpu": "cuda:0",
              "seed": 72, "training_shape": X.shape, "query_shape": Q.shape,
              "input_source": str(args.inputs) if args.inputs else "synthetic_smoke_not_formal",
              "n_ensembles": 8,
              "context_size": _tabdpt_context_size(args.model) if args.model.startswith("tabdpt") else None,
              "atol": 1e-4, "rtol": 1e-4, "fit_mode": args.fit_mode,
              "version": version("tabpfn" if args.model == "tabpfn" else "tabdpt"),
              "input_selection": input_metadata,
              "batches": []}
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
                uses_predict_batch_size = args.model in {"tabdpt120", "tabdpt130"}
                context_size = _tabdpt_context_size(args.model)
                if not uses_predict_batch_size:
                    options["inf_batch_size"] = batch
                model = TabDPTRegressor(**options)
                model.fit(X, y)
                predict = partial(_predict_tabdpt, model, Q, batch,
                                  uses_predict_batch_size, context_size)
                record["actual_context_rows"] = (
                    len(X) if context_size is None else min(len(X), context_size)
                )
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
