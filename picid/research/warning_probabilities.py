"""Probability conversions for XJTU HI forecasts and fixed-unit warning events."""

from __future__ import annotations

import numpy as np


def tabdpt_distribution_mean(logits: np.ndarray, borders: np.ndarray) -> np.ndarray:
    """Integrate TabDPT's piecewise-uniform predictive histogram."""
    scores = np.asarray(logits, dtype=np.float64)
    edges = np.asarray(borders, dtype=np.float64).reshape(-1)
    if scores.ndim != 2 or scores.shape[1] + 1 != len(edges):
        raise ValueError(
            "logits must be (samples, bins) with one more border than bins"
        )
    if (
        not np.isfinite(scores).all()
        or not np.isfinite(edges).all()
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("logits and strictly increasing borders must be finite")
    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return probabilities @ centers


def tabdpt_distribution_cdf(
    logits: np.ndarray, borders: np.ndarray, threshold: float
) -> np.ndarray:
    """Evaluate a continuous CDF by integrating uniform probability within bins."""
    scores = np.asarray(logits, dtype=np.float64)
    edges = np.asarray(borders, dtype=np.float64).reshape(-1)
    if scores.ndim != 2 or scores.shape[1] + 1 != len(edges):
        raise ValueError(
            "logits must be (samples, bins) with one more border than bins"
        )
    if (
        not np.isfinite(scores).all()
        or not np.isfinite(edges).all()
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("logits and strictly increasing borders must be finite")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    mass_fraction = np.clip((float(threshold) - edges[:-1]) / np.diff(edges), 0.0, 1.0)
    return probabilities @ mass_fraction


def residual_ecdf_score(
    residuals_by_device: dict[str, np.ndarray],
    predicted_mean: np.ndarray,
    threshold: float,
    *,
    exclude_device: str | None = None,
) -> np.ndarray:
    """Estimate event probability from equally weighted device residual ECDFs."""
    usable = {
        device: np.asarray(values, dtype=np.float64).reshape(-1)
        for device, values in residuals_by_device.items()
        if device != exclude_device
    }
    if not usable or any(not np.isfinite(values).all() for values in usable.values()):
        raise ValueError(
            "Residual calibration requires finite samples from at least one device"
        )
    query = np.asarray(predicted_mean, dtype=np.float64).reshape(-1)
    if not np.isfinite(query).all() or not np.isfinite(threshold):
        raise ValueError("predicted means and warning threshold must be finite")
    limit = float(threshold) - query
    per_device = [
        np.searchsorted(np.sort(values), limit, side="right") / len(values)
        for values in usable.values()
    ]
    return np.mean(np.stack(per_device, axis=0), axis=0)
