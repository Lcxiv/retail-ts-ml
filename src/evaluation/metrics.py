"""
Forecast Evaluation Metrics.

All metrics handle zero-safe division and NaN in actuals/predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove NaN pairs."""
    mask = ~(np.isnan(a) | np.isnan(b))
    return a[mask], b[mask]


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    a, p = _safe(np.asarray(actual, float), np.asarray(predicted, float))
    return float(np.mean(np.abs(a - p)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    a, p = _safe(np.asarray(actual, float), np.asarray(predicted, float))
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-6) -> float:
    """Mean Absolute Percentage Error. Skip zeros in actual."""
    a, p = _safe(np.asarray(actual, float), np.asarray(predicted, float))
    mask = np.abs(a) > eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


def smape(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-6) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    a, p = _safe(np.asarray(actual, float), np.asarray(predicted, float))
    denom = (np.abs(a) + np.abs(p)) / 2 + eps
    return float(np.mean(np.abs(a - p) / denom) * 100)


def wape(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-6) -> float:
    """Weighted Absolute Percentage Error."""
    a, p = _safe(np.asarray(actual, float), np.asarray(predicted, float))
    return float(np.sum(np.abs(a - p)) / (np.sum(np.abs(a)) + eps) * 100)


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean forecast bias (positive = over-prediction)."""
    a, p = _safe(np.asarray(actual, float), np.asarray(predicted, float))
    return float(np.mean(p - a))


def all_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute all metrics and return as a dict."""
    return {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "MAPE": mape(actual, predicted),
        "sMAPE": smape(actual, predicted),
        "WAPE": wape(actual, predicted),
        "Bias": bias(actual, predicted),
    }


def evaluate_by_segment(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    segment_cols: list[str],
) -> pd.DataFrame:
    """
    Evaluate metrics grouped by segment columns.

    Returns a DataFrame with one row per segment and one column per metric.
    """
    rows = []
    groups = df.groupby(segment_cols)
    for key, grp in groups:
        metrics = all_metrics(grp[actual_col].to_numpy(), grp[pred_col].to_numpy())
        row = dict(zip(segment_cols, key if isinstance(key, tuple) else [key]))
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)
