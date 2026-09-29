"""Device-aware probability calibration and alarm metrics for PHMD studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


def warning_labels_from_rul(
    rul: np.ndarray, horizons: dict[str, float]
) -> dict[str, np.ndarray]:
    """Build warning labels from observed PHMD RUL values in native intervals."""
    values = np.asarray(rul, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("RUL labels must be non-empty and finite")
    if not horizons or any(not np.isfinite(h) or h <= 0 for h in horizons.values()):
        raise ValueError("Warning horizons must be positive finite intervals")
    return {name: values <= float(horizon) for name, horizon in horizons.items()}


def _device_equal_weights(unit_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(unit_ids).reshape(-1)
    if ids.size == 0:
        raise ValueError("unit_ids must be non-empty")
    weights = np.empty(ids.size, dtype=np.float64)
    unique, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    del unique
    weights[:] = 1.0 / counts[inverse]
    weights *= len(counts) / weights.sum()
    return weights


def _score_column(scores: np.ndarray, *, probability: bool) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Calibration scores must be finite")
    if probability:
        eps = np.finfo(np.float64).eps
        values = np.log(np.clip(values, eps, 1 - eps) / np.clip(1 - values, eps, 1))
    return values.reshape(-1, 1)


def fit_platt_calibrator(
    scores: np.ndarray,
    labels: np.ndarray,
    unit_ids: np.ndarray,
    *,
    score_is_probability: bool,
) -> LogisticRegression:
    """Fit a weighted Platt map with equal total calibration weight per device."""
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    ids = np.asarray(unit_ids).reshape(-1)
    x = _score_column(scores, probability=score_is_probability)
    if not (len(x) == len(y) == len(ids)):
        raise ValueError("scores, labels, and unit_ids must have equal lengths")
    if not np.isin(y, [0, 1]).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError("Calibration requires positive and negative examples")
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(x, y, sample_weight=_device_equal_weights(ids))
    return model


def apply_platt_calibrator(
    model: LogisticRegression,
    scores: np.ndarray,
    *,
    score_is_probability: bool,
) -> np.ndarray:
    """Map a probability-like score or logit to calibrated event probabilities."""
    return model.predict_proba(_score_column(scores, probability=score_is_probability))[
        :, 1
    ]


def select_empirical_fpr_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    unit_ids: np.ndarray,
    *,
    target_fpr: float = 0.05,
) -> tuple[float, float]:
    """Choose a score threshold from calibration negatives only.

    The returned empirical FPR is descriptive. Correlated time samples and the
    small number of calibration devices do not support a population guarantee.
    """
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=bool).reshape(-1)
    ids = np.asarray(unit_ids).reshape(-1)
    if not (len(score) == len(y) == len(ids)):
        raise ValueError("scores, labels, and unit_ids must have equal lengths")
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must lie strictly between zero and one")
    negative = ~y
    if not negative.any():
        raise ValueError("Calibration set has no negative warning examples")
    if not np.isfinite(score).all():
        raise ValueError("Calibration scores must be finite")
    # Select against the bearing-macro FPR. A pooled timestamp quantile would
    # let the longest calibration bearing dominate this deployable threshold.
    candidates = np.r_[
        np.unique(score[negative]), np.nextafter(score[negative].max(), np.inf)
    ]
    candidates.sort()
    valid_fprs = []
    for candidate in candidates:
        fprs = []
        for unit in np.unique(ids):
            mask = (ids == unit) & negative
            if mask.any():
                fprs.append(float(np.mean(score[mask] >= candidate)))
        macro_fpr = float(np.mean(fprs))
        if macro_fpr <= target_fpr:
            valid_fprs.append((float(candidate), macro_fpr))
    # Highest recall subject to the empirical device-macro FPR constraint.
    if not valid_fprs:
        return float(np.nextafter(score[negative].max(), np.inf)), 0.0
    return min(valid_fprs, key=lambda item: item[0])


def probability_metrics(
    probabilities: np.ndarray, labels: np.ndarray, unit_ids: np.ndarray
) -> dict[str, float | int]:
    """Return event scores with equal bearing weight and device bootstrap inputs."""
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    ids = np.asarray(unit_ids).reshape(-1)
    if not (len(p) == len(y) == len(ids)) or not len(p):
        raise ValueError(
            "probabilities, labels, and unit_ids must align and be non-empty"
        )
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite values in [0, 1]")
    eps = np.finfo(np.float64).eps
    p_clip = np.clip(p, eps, 1 - eps)
    per_device_brier: list[float] = []
    per_device_logloss: list[float] = []
    for unit in np.unique(ids):
        mask = ids == unit
        per_device_brier.append(float(np.mean((p[mask] - y[mask]) ** 2)))
        per_device_logloss.append(
            float(
                np.mean(
                    -(
                        y[mask] * np.log(p_clip[mask])
                        + (1 - y[mask]) * np.log(1 - p_clip[mask])
                    )
                )
            )
        )
    if np.unique(y).size == 2:
        pr_auc = float(
            average_precision_score(y, p, sample_weight=_device_equal_weights(ids))
        )
    else:
        pr_auc = float("nan")
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (p >= lower) & ((p < upper) if upper < 1 else (p <= upper))
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(p[mask].mean()) - float(y[mask].mean())
            )
    return {
        "device_count": int(np.unique(ids).size),
        "query_count": int(len(y)),
        "prevalence": float(y.mean()),
        "device_macro_brier": float(np.mean(per_device_brier)),
        "device_macro_log_loss": float(np.mean(per_device_logloss)),
        "device_weighted_auprc": pr_auc,
        "query_weighted_ece_10bin": float(ece),
    }


@dataclass(frozen=True)
class AlarmMetrics:
    """Device-level detection, nuisance alarms, and lead-time summaries."""

    metrics: dict[str, Any]
    per_device: dict[str, dict[str, Any]]


def alarm_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    rul_at_query: np.ndarray,
    unit_ids: np.ndarray,
    query_times: np.ndarray,
    *,
    threshold: float,
) -> AlarmMetrics:
    """Evaluate fixed-threshold alarm episodes on ordered bearing trajectories.

    Consecutive positive queries are one alarm episode. Episodes beginning
    before the warning window are false alarms; only episodes initiated inside
    it count as detections, even if an earlier alarm persists into the window.
    Test RUL is used only to construct the offline event label
    and describe lead time, never to construct a model score or threshold.
    """
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=bool).reshape(-1)
    rul = np.asarray(rul_at_query, dtype=np.float64).reshape(-1)
    ids = np.asarray(unit_ids).reshape(-1)
    times = np.asarray(query_times, dtype=np.int64).reshape(-1)
    if not (len(score) == len(y) == len(rul) == len(ids) == len(times)):
        raise ValueError("alarm arrays must have equal lengths")
    if not np.isfinite(score).all() or not np.isfinite(rul).all():
        raise ValueError("scores and RUL must be finite")
    per_device: dict[str, dict[str, Any]] = {}
    false_episode_counts: list[int] = []
    fpr_by_device: list[float] = []
    detection: list[float] = []
    leads: list[float] = []
    all_alarm_counts: list[int] = []
    for unit in np.unique(ids):
        mask = ids == unit
        order = np.argsort(times[mask], kind="stable")
        local_scores = score[mask][order]
        local_y = y[mask][order]
        local_rul = rul[mask][order]
        local_times = times[mask][order]
        local_alarm = local_scores >= threshold
        starts = local_alarm & np.r_[True, ~local_alarm[:-1]]
        false_starts = starts & ~local_y
        event_starts = starts & local_y
        detected = bool(event_starts.any())
        if detected:
            first_event_alarm = int(np.flatnonzero(event_starts)[0])
            lead = float(local_rul[first_event_alarm])
            leads.append(lead)
        else:
            lead = None
        negatives = ~local_y
        fpr = (
            float(np.mean(local_alarm[negatives])) if negatives.any() else float("nan")
        )
        false_count = int(false_starts.sum())
        alarm_count = int(starts.sum())
        key = str(unit)
        per_device[key] = {
            "query_count": int(mask.sum()),
            "warning_query_count": int(local_y.sum()),
            "detected_in_warning_window": detected,
            "missed_warning_window": not detected,
            "false_alarm_query_fpr": fpr,
            "false_alarm_episode_count": false_count,
            "alarm_episode_count": alarm_count,
            "lead_time_intervals": lead,
            "first_warning_interval": (
                int(local_times[np.flatnonzero(event_starts)[0]]) if detected else None
            ),
        }
        false_episode_counts.append(false_count)
        fpr_by_device.append(fpr)
        detection.append(float(detected))
        all_alarm_counts.append(alarm_count)
    valid_fprs = [v for v in fpr_by_device if np.isfinite(v)]
    metrics = {
        "device_count": int(len(per_device)),
        "detection_rate": float(np.mean(detection)),
        "missed_device_rate": float(1 - np.mean(detection)),
        "mean_device_false_alarm_query_fpr": float(np.mean(valid_fprs))
        if valid_fprs
        else float("nan"),
        "mean_false_alarm_episodes_per_device": float(np.mean(false_episode_counts)),
        "mean_alarm_episodes_per_device": float(np.mean(all_alarm_counts)),
        "detected_device_lead_time_intervals": {
            "count": len(leads),
            "mean": float(np.mean(leads)) if leads else float("nan"),
            "median": float(np.median(leads)) if leads else float("nan"),
            "values": leads,
        },
        "threshold": float(threshold),
    }
    return AlarmMetrics(metrics=metrics, per_device=per_device)
