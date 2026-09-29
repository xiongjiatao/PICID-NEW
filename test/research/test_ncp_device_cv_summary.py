import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.summarize_ncp_device_cv import SEEDS, aggregate


CANDIDATES = ((1, 1), (5, 1), (10, 5), (20, 5), (50, 50))


def _write_complete_fixture(run_root: Path) -> None:
    for seed in SEEDS:
        for fold in range(5):
            fold_dir = run_root / f"seed{seed}" / f"fold{fold}"
            fold_dir.mkdir(parents=True)
            held_out = [f"DS{source:02d}-unit{fold + 1:02d}" for source in (1, 4, 5, 7)]
            unit_ids = np.repeat(np.asarray(held_out), 2)
            query_times = np.tile(np.arange(2, dtype=np.int64), len(held_out))
            targets = np.arange(len(unit_ids), dtype=np.float64)
            query_counts = {device: 2 for device in held_out}
            candidates = []
            for window, stride in CANDIDATES:
                filename = f"fold{fold}_window{window}_stride{stride}_predictions.npz"
                prediction_path = fold_dir / filename
                np.savez_compressed(
                    prediction_path,
                    predictions_normalized=np.zeros(len(unit_ids)),
                    targets_native=targets,
                    targets_normalized=targets,
                    unit_ids=unit_ids,
                    query_times=query_times,
                )
                per_device = {
                    device: {
                        "mae": 1.0,
                        "mse": 1.0,
                        "rmse": 1.0,
                        "nasa_score": 1.0,
                        "5pct_mae": 1.0,
                        "10pct_mae": 1.0,
                        "20pct_mae": 1.0,
                    }
                    for device in held_out
                }
                candidates.append(
                    {
                        "window": window,
                        "stride": stride,
                        "metrics_native_rul": {"per_device": per_device},
                        "query_rows_by_device": query_counts,
                        "prediction_sha256": hashlib.sha256(
                            prediction_path.read_bytes()
                        ).hexdigest(),
                        "fit_seconds": 1.0,
                        "predict_seconds": 2.0,
                        "data_preparation_seconds": 0.1,
                        "model_initialization_seconds": 0.2,
                        "serialization_seconds": 0.1,
                        "digest_seconds": 0.1,
                        "candidate_wall_seconds": 3.5,
                        "query_throughput_per_second": 4.0,
                        "peak_allocated_gib": 0.5,
                        "peak_reserved_gib": 1.0,
                    }
                )
            manifest = {
                "status": "complete",
                "seed": seed,
                "fold": fold,
                "wall_seconds": 18.0,
                "fold_devices": {"held_out": held_out},
                "query_protocol": {
                    "rule": "every transformed row in chronological order; no test/query subsampling",
                    "indices_by_device": {device: {"count": 2} for device in held_out},
                    "total_rows": len(unit_ids),
                },
                "candidates": candidates,
            }
            (fold_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )


def test_summary_accepts_full_query_predictions_and_counts_all_costs(tmp_path):
    run_root = tmp_path / "runs"
    _write_complete_fixture(run_root)

    summary = aggregate(run_root, SEEDS)

    assert summary["held_out_device_count"] == 20
    assert summary["compute_cost"]["candidate_evaluations"] == 125
    assert summary["compute_cost"]["candidates"]["w1_s1"]["total_query_rows"] == 200
    assert summary["seed_mean_and_sd"]["w1_s1"]["mae"]["mean_across_seeds"] == 1.0


def test_summary_rejects_reordered_or_missing_timeline_rows(tmp_path):
    run_root = tmp_path / "runs"
    _write_complete_fixture(run_root)
    fold_dir = run_root / "seed72" / "fold0"
    prediction_path = fold_dir / "fold0_window1_stride1_predictions.npz"
    with np.load(prediction_path, allow_pickle=False) as prediction:
        arrays = {key: np.asarray(prediction[key]) for key in prediction.files}
    arrays["query_times"][[0, 1]] = arrays["query_times"][[1, 0]]
    np.savez_compressed(prediction_path, **arrays)
    manifest_path = fold_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["candidates"][0]["prediction_sha256"] = hashlib.sha256(
        prediction_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    with pytest.raises(ValueError, match="incomplete, repeated, or reordered"):
        aggregate(run_root, SEEDS)
