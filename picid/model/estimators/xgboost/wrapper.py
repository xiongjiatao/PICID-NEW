"""Fit/predict wrappers backed by the XGBoost library."""

from pathlib import Path
from typing import override

import joblib
import numpy as np
import torch
from xgboost import XGBClassifier, XGBRegressor

from picid.model.adapters.base import AbstractFitPredictWrapper
from picid.model.definitions import CLASSIFICATION_TASKS, REGRESSION_TASKS


class FitPredictXGBoostWrapper(AbstractFitPredictWrapper):
    """Wrap XGBoost estimators in PICID's fit/predict interface.

    ``n_estimators=1000`` and ``random_state=42`` retain the explicit defaults
    in the released PICID wrapper. Other tree parameters are XGBoost 3.1.3
    defaults unless specified in Hydra configuration. The TFM-PHM paper gives
    context/stride candidates but does not enumerate its tree-parameter search
    grid, so these defaults are a transparent reproduction assumption.
    """

    def __init__(
        self,
        task_type: str,
        n_estimators: int = 1000,
        model_cache_path: str | None = None,
        random_state: int | None = 42,
        num_classes: int | None = None,
        n_jobs: int = 1,
        device: str = "cpu",
        seq_len: int | None = None,
        label_len: int | None = None,
        pred_len: int | None = None,
        features_mode: str | None = None,
        **xgb_params,
    ):
        self.regression_tasks = REGRESSION_TASKS
        self.classification_tasks = CLASSIFICATION_TASKS
        self.supported_types = self.regression_tasks + self.classification_tasks
        self.task_type = task_type
        self.num_classes = num_classes

        if task_type not in self.supported_types:
            raise ValueError(f"Task {task_type} not supported for XGBoost.")

        if task_type in self.classification_tasks and num_classes is None:
            raise ValueError("num_classes must be provided for classification tasks.")

        estimator_params = {
            "n_estimators": n_estimators,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "device": device,
            "verbosity": 0,
        }
        estimator_params.update(xgb_params)

        if task_type in self.classification_tasks:
            estimator_params.setdefault(
                "objective",
                "multi:softprob" if num_classes > 2 else "binary:logistic",
            )
            backbone = XGBClassifier(**estimator_params)
        else:
            estimator_params.setdefault("objective", "reg:squarederror")
            backbone = XGBRegressor(**estimator_params)

        self.model_cache_path = model_cache_path
        super().__init__(
            backbone=backbone,
            task_type=task_type,
            num_classes=num_classes,
        )

    @override
    def _call_fit(self, X: torch.Tensor, y: torch.Tensor):
        """Convert PICID tensors to the one-dimensional XGBoost label API."""
        X_array = X.detach().cpu().numpy()
        y_array = y.detach().cpu().numpy().reshape(-1)
        self.backbone.fit(X_array, y_array)

    @override
    def _call_predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return regression predictions or class probabilities as a tensor."""
        X_array = X.detach().cpu().numpy()
        if self.task_type in CLASSIFICATION_TASKS:
            predictions = self.backbone.predict_proba(X_array)
        else:
            predictions = self.backbone.predict(X_array)
        return torch.as_tensor(np.asarray(predictions), dtype=torch.float32)

    @override
    def serialize_model(self, task_id: str | None = None) -> str:
        """Serialize the fitted estimator beneath the configured model cache."""
        if task_id is None:
            raise ValueError("No model_path provided in kwargs and no task_id given.")
        model_dir = (
            Path(".model_cache")
            if self.model_cache_path is None
            else Path(self.model_cache_path)
        )
        model_path = model_dir / f"{task_id}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.backbone, model_path)
        return str(model_path)

    @override
    def load_model(self, task_id: str | None = None):
        """Load a serialized estimator from the configured model cache."""
        if task_id is None:
            raise ValueError("No model_path provided in kwargs and no task_id given.")
        model_dir = (
            Path(".model_cache")
            if self.model_cache_path is None
            else Path(self.model_cache_path)
        )
        model_path = model_dir / f"{task_id}.joblib"
        self.backbone = joblib.load(model_path)
        return self.backbone

    @property
    @override
    def allows_multi_target(self) -> bool:
        """XGBoost wrapper currently exposes one target per fit-predict task."""
        return False


__all__ = ["FitPredictXGBoostWrapper"]
