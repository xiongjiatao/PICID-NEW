import numpy as np
import pytest

from picid.research.chunks import predict_chunks
from picid.research.protocol import candidates, freeze_selection


def test_full_grid_and_no_candidate_test_access():
    grid = list(candidates())
    assert len(grid) == 58
    assert len({c.key for c in grid}) == 58
    assert all("test=false" in c.overrides() for c in grid)
    xgb_pairs = {(c.window, c.stride) for c in grid if c.dataset == "nc_p" and c.model == "xgboost"}
    assert xgb_pairs == {(1, 1), (5, 1), (10, 5), (20, 5), (50, 50)}


def test_selection_rejects_missing_failed_or_test_accessed_results():
    grid = [c for c in candidates() if c.dataset == "nc_p" and c.model == "xgboost"]
    results = {c.key: dict(status="success", seed=72, test_enabled=False,
                          test_metrics_present=False, val_loss=float(i)) for i, c in enumerate(grid)}
    assert freeze_selection(grid, results)["candidate"]["window"] == 1
    results[grid[0].key]["test_metrics_present"] = True
    with pytest.raises(ValueError):
        freeze_selection(grid, results)
    with pytest.raises(ValueError):
        freeze_selection(grid, {})


def test_interrupted_resume_is_complete_and_ordered(tmp_path):
    X = np.arange(30).reshape(10, 3)
    identity = dict(config="c", weights="w", training_input="t", code="v", physical_gpus=[0])
    calls = []
    def interrupted(x):
        calls.append(x[0, 0])
        if len(calls) == 2:
            raise RuntimeError("interruption")
        return x[:, 0]
    with pytest.raises(RuntimeError):
        predict_chunks(interrupted, X, tmp_path, identity, 3)
    calls.clear()
    def resume(x):
        calls.append(x[0, 0])
        return x[:, 0]
    np.testing.assert_array_equal(predict_chunks(resume, X, tmp_path, identity, 3), X[:, 0])
    assert calls == [9, 18, 27]
    with pytest.raises(ValueError, match="fingerprint"):
        predict_chunks(resume, X[::-1], tmp_path, identity, 3)
    chunk = next(tmp_path.glob("*.npy"))
    chunk.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        predict_chunks(resume, X, tmp_path, identity, 3)
