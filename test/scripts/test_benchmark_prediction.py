import numpy as np

from scripts.research.benchmark_prediction import _predict_tabdpt, _tabdpt_context_size


def test_tabdpt120_benchmark_preserves_native_context_and_batch_api():
    calls = []

    class Model:
        def predict(self, queries, **kwargs):
            calls.append(kwargs)
            return np.zeros(len(queries))

    context_size = _tabdpt_context_size("tabdpt120")
    assert context_size is None
    _predict_tabdpt(Model(), np.zeros((5, 2)), 32, True, context_size)
    assert calls == [{"n_ensembles": 8, "seed": 72, "batch_size": 32}]


def test_older_tabdpt_benchmark_retains_explicit_context_cap():
    assert _tabdpt_context_size("tabdpt") == 2048
    assert _tabdpt_context_size("tabdpt130") == 2048
