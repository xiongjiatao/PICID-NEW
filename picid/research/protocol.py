"""Frozen candidate grid and validation-only selection rules."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

SEEDS = (72, 88, 101)
PHYSICAL_GPUS = (0, 1, 2, 3, 4, 5)
WINDOWS = ((1, 1), (5, 1), (10, 5), (20, 5), (50, 50))
DATASETS = {
    "nc_p": "concepts_n_cmapss_multi/prognostics",
    "xjtu": "xjtu_sy/prognostics/phmd_split/combined",
}
MODELS = ("tabdpt", "tabdpt130", "tabpfn", "xgboost", "lstm")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Candidate:
    dataset: str
    model: str
    window: int
    stride: int
    lr: float | None = None

    @property
    def key(self):
        return f"{self.dataset}_{self.model}_w{self.window}_s{self.stride}_lr{self.lr}"

    def overrides(self, seed=72, test=False):
        if seed not in SEEDS:
            raise ValueError("Unregistered seed")
        base_model = "tabdpt" if self.model == "tabdpt130" else self.model
        suffix = "lstm" if base_model == "lstm" else f"{base_model}_fit_predict"
        args = [f"experiment={DATASETS[self.dataset]}/{suffix}", f"seed={seed}",
                f"test={str(test).lower()}", f"task_definition.seq_len={self.window}",
                f"task_definition.stride_train={self.stride}",
                "task_definition.subset_seed=72", "logger=csv", "num_threads=8"]
        if self.model == "tabdpt130":
            args.append("model=tabdpt130_fit_predict")
        if self.model == "lstm":
            args += ["trainer.max_epochs=200", "datamodule.train_batch_size=512",
                     "datamodule.val_batch_size=1024", "datamodule.test_batch_size=1024",
                     f"optimization.lr={self.lr}"]
        return args


def candidates():
    for dataset in DATASETS:
        for model in MODELS:
            if model == "lstm":
                for window in (1, 10, 50):
                    for lr in (1e-3, 5e-4, 1e-4):
                        yield Candidate(dataset, model, window, 1, lr)
            else:
                for window, stride in WINDOWS:
                    yield Candidate(dataset, model, window, stride)


def freeze_selection(grid, results, execution_overrides=()):
    """Fail closed on missing/failed candidates or accidental test evaluation."""
    expected = {c.key: c for c in grid}
    if set(results) != set(expected) or not expected:
        raise ValueError("All registered candidates must finish before selection")
    if len({(c.dataset, c.model) for c in grid}) != 1:
        raise ValueError("Selection must contain one dataset/model family")
    for result in results.values():
        if (result["status"] != "success" or result["seed"] != 72
                or result["test_enabled"]
                or result.get("test_metrics_present", result.get("test_fields_populated", False))
                or not math.isfinite(result.get("val_loss", result.get("best_val_loss", math.nan)))):
            raise ValueError("Invalid validation-only result")
    winner = min(expected, key=lambda key: (
        results[key].get("val_loss", results[key].get("best_val_loss", math.inf)), key
    ))
    payload = {"protocol": "three_seed_fixed_configuration", "seeds": SEEDS,
               "candidate": asdict(expected[winner]),
               "execution_overrides": list(execution_overrides),
               "selection_results": results}
    return {**payload, "sha256": digest(payload)}
