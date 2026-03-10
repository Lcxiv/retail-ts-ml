"""
Baseline Forecasting Models.

Always benchmark against these simple baselines before training advanced models.
All models follow a common predict(train, horizon) interface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Protocol


class Forecaster(Protocol):
    """Common interface for all forecasters."""
    def fit(self, series: pd.Series) -> None: ...
    def predict(self, horizon: int) -> np.ndarray: ...
    name: str


class NaiveForecaster:
    """Last-value naive forecast."""
    name = "naive"

    def fit(self, series: pd.Series) -> None:
        self._last = series.dropna().iloc[-1]

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._last)


class SeasonalNaiveForecaster:
    """
    Seasonal naive: carry forward the value from the same period last season.
    Default season_length=52 for weekly data (same week last year).
    """
    name = "seasonal_naive"

    def __init__(self, season_length: int = 52):
        self.season_length = season_length
        self._history: np.ndarray | None = None

    def fit(self, series: pd.Series) -> None:
        self._history = series.dropna().to_numpy()

    def predict(self, horizon: int) -> np.ndarray:
        preds = []
        h = self._history
        for step in range(1, horizon + 1):
            idx = -self.season_length + ((step - 1) % self.season_length)
            if abs(idx) > len(h):
                preds.append(h[-1])  # fallback
            else:
                preds.append(h[idx])
        return np.array(preds)


class MovingAverageForecaster:
    """Rolling moving average forecast."""
    name = "moving_average"

    def __init__(self, window: int = 4):
        self.window = window
        self._mean: float | None = None

    def fit(self, series: pd.Series) -> None:
        clean = series.dropna()
        self._mean = float(clean.iloc[-self.window:].mean())

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._mean)


class SimpleExponentialSmoothing:
    """
    Simple exponential smoothing (SES) with manual alpha.
    SES: s_t = alpha * y_t + (1 - alpha) * s_{t-1}
    """
    name = "ses"

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._level: float | None = None

    def fit(self, series: pd.Series) -> None:
        clean = series.dropna().to_numpy(dtype=float)
        s = clean[0]
        for y in clean[1:]:
            s = self.alpha * y + (1 - self.alpha) * s
        self._level = s

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._level)


ALL_BASELINES = [
    NaiveForecaster,
    SeasonalNaiveForecaster,
    MovingAverageForecaster,
    SimpleExponentialSmoothing,
]


def run_all_baselines(
    train_series: pd.Series,
    horizon: int,
) -> dict[str, np.ndarray]:
    """Fit all baseline models and return forecasts keyed by model name."""
    results = {}
    for ModelClass in ALL_BASELINES:
        model = ModelClass()
        model.fit(train_series)
        results[model.name] = model.predict(horizon)
    return results
