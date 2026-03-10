"""Rolling-Origin Backtesting Engine.

Simulates future forecasting by training on progressively larger windows
and predicting a fixed horizon. This is the correct time-aware validation method.
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from src.evaluation.metrics import all_metrics

log = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    date_col: str = "date"
    target_col: str = "qty_sold"
    horizon: int = 13          # forecast horizon in periods
    n_splits: int = 4          # number of rolling folds
    min_train_periods: int = 52  # minimum training length in periods


@dataclass
class BacktestResult:
    fold: int
    train_end: pd.Timestamp
    actuals: np.ndarray
    predictions: np.ndarray
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.metrics = all_metrics(self.actuals, self.predictions)


def rolling_origin_backtest(
    df: pd.DataFrame,
    forecast_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    config: BacktestConfig | None = None,
) -> list[BacktestResult]:
    """
    Perform rolling-origin cross-validation.

    Parameters
    ----------
    df : DataFrame sorted by date (all series combined, long format)
    forecast_fn : function(train_df, test_df) -> np.ndarray of predictions
                  Must be implemented by the caller (model-agnostic).
    config : BacktestConfig

    Returns
    -------
    list of BacktestResult, one per fold
    """
    if config is None:
        config = BacktestConfig()

    df = df.sort_values(config.date_col).copy()
    dates = df[config.date_col].unique()
    dates = np.sort(dates)
    n = len(dates)

    # Calculate fold cutpoints (exclude the last horizon periods from training)
    usable = n - config.horizon
    if usable < config.min_train_periods + config.n_splits:
        raise ValueError("Not enough data for the requested number of splits and horizon.")

    step = (usable - config.min_train_periods) // config.n_splits
    cutpoints = [
        dates[config.min_train_periods + i * step - 1]
        for i in range(1, config.n_splits + 1)
    ]

    results = []
    for fold_idx, cutpoint in enumerate(cutpoints):
        train_df = df[df[config.date_col] <= cutpoint]
        test_dates_start = cutpoint + pd.Timedelta(days=1)
        test_end = dates[min(np.searchsorted(dates, cutpoint) + config.horizon, n - 1)]
        test_df = df[(df[config.date_col] > cutpoint) & (df[config.date_col] <= test_end)]

        if test_df.empty:
            log.warning("Fold %d: empty test set, skipping", fold_idx + 1)
            continue

        predictions = forecast_fn(train_df, test_df)
        actuals = test_df[config.target_col].fillna(0).to_numpy()

        result = BacktestResult(
            fold=fold_idx + 1,
            train_end=pd.Timestamp(cutpoint),
            actuals=actuals,
            predictions=predictions,
        )
        log.info(
            "Fold %d | train_end=%s | test_rows=%d | MAE=%.2f | RMSE=%.2f",
            result.fold, result.train_end.date(),
            len(actuals), result.metrics["MAE"], result.metrics["RMSE"]
        )
        results.append(result)

    return results


def summarize_backtest(results: list[BacktestResult]) -> pd.DataFrame:
    """Summarize backtest fold metrics into a single DataFrame."""
    rows = []
    for r in results:
        row = {"fold": r.fold, "train_end": r.train_end, "n_test": len(r.actuals)}
        row.update(r.metrics)
        rows.append(row)
    return pd.DataFrame(rows)
