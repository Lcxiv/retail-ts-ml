"""Feature Engineering Pipeline.

All features are computed in a time-aware manner — no future leakage.
Adjust lag sizes and rolling windows based on the confirmed grain.
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np

from src.config.config import (
    DATE_COLUMN, RETAILER_COLUMN, STORE_COLUMN, PRODUCT_COLUMN,
    PRIMARY_TARGET, DEFAULT_GRAIN,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------

WEEKLY_LAGS = [1, 2, 4, 8, 12, 26, 52]
MONTHLY_LAGS = [1, 2, 3, 6, 12]


def add_lag_features(
    df: pd.DataFrame,
    target: str = PRIMARY_TARGET,
    lags: list[int] | None = None,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Add lag features within each (retailer, store, product) group.

    Parameters
    ----------
    df : DataFrame sorted by [entity_cols, date_col]
    target : column to lag
    lags : lag periods (in units of the time grain)
    group_cols : columns that define a unique time series
    """
    if lags is None:
        lags = WEEKLY_LAGS if DEFAULT_GRAIN == "weekly" else MONTHLY_LAGS
    if group_cols is None:
        group_cols = [RETAILER_COLUMN, STORE_COLUMN, PRODUCT_COLUMN]

    df = df.sort_values(group_cols + [DATE_COLUMN]).copy()
    grp = df.groupby(group_cols, sort=False)[target]

    for lag in lags:
        df[f"lag_{lag}"] = grp.shift(lag)

    log.info("Added %d lag features", len(lags))
    return df


# ---------------------------------------------------------------------------
# Rolling window features
# ---------------------------------------------------------------------------

def add_rolling_features(
    df: pd.DataFrame,
    target: str = PRIMARY_TARGET,
    windows: list[int] | None = None,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Add rolling mean, std, max, min, and growth rate features."""
    if windows is None:
        windows = [4, 8, 12]
    if group_cols is None:
        group_cols = [RETAILER_COLUMN, STORE_COLUMN, PRODUCT_COLUMN]

    df = df.sort_values(group_cols + [DATE_COLUMN]).copy()

    for w in windows:
        grp = df.groupby(group_cols, sort=False)[target]
        df[f"rolling_mean_{w}"] = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        df[f"rolling_std_{w}"] = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=2).std())
        df[f"rolling_max_{w}"] = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
        df[f"rolling_min_{w}"] = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).min())

    # Growth rate: (current_lag1 - lag(w+1)) / lag(w+1)
    for w in [4, 8]:
        lag1 = df.groupby(group_cols, sort=False)[target].shift(1)
        lag_w = df.groupby(group_cols, sort=False)[target].shift(w + 1)
        df[f"rolling_growth_{w}"] = (lag1 - lag_w) / (lag_w.abs() + 1e-6)

    log.info("Added rolling features for windows: %s", windows)
    return df


# ---------------------------------------------------------------------------
# Time / calendar features
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-derived features from the date column."""
    df = df.copy()
    dt = pd.to_datetime(df[DATE_COLUMN])
    df["year"] = dt.dt.year
    df["quarter"] = dt.dt.quarter
    df["month"] = dt.dt.month
    df["week"] = dt.dt.isocalendar().week.astype(int)
    df["day_of_year"] = dt.dt.dayofyear
    df["week_index"] = (dt - dt.min()).dt.days // 7  # monotonic time index
    df["is_q4"] = (df["quarter"] == 4).astype(int)
    df["is_holiday_season"] = (df["month"].isin([11, 12])).astype(int)
    log.info("Added time features")
    return df


# ---------------------------------------------------------------------------
# Comparative / normalized features
# ---------------------------------------------------------------------------

def add_comparative_features(
    df: pd.DataFrame,
    target: str = PRIMARY_TARGET,
) -> pd.DataFrame:
    """
    Add retailer share, store share, z-score, and normalized index.
    These are safe to compute per-period (no future leakage).
    """
    df = df.copy()

    # Retailer share: store's share of retailer total in that period
    period_retailer_total = df.groupby([DATE_COLUMN, RETAILER_COLUMN])[target].transform("sum")
    df["store_share_of_retailer"] = df[target] / (period_retailer_total + 1e-6)

    # Store's z-score vs its own historical mean/std
    store_mean = df.groupby(STORE_COLUMN)[target].transform("mean")
    store_std = df.groupby(STORE_COLUMN)[target].transform("std").fillna(1)
    df["store_zscore"] = (df[target] - store_mean) / (store_std + 1e-6)

    log.info("Added comparative features")
    return df


# ---------------------------------------------------------------------------
# Hierarchical aggregate features
# ---------------------------------------------------------------------------

def add_hierarchical_features(
    df: pd.DataFrame,
    target: str = PRIMARY_TARGET,
    hierarchy_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Add category-level and department-level average trends.
    These aggregate from lag-shifted values to avoid leakage.
    """
    if hierarchy_cols is None:
        hierarchy_cols = ["category", "department"]

    df = df.copy()
    lag_col = "lag_1" if "lag_1" in df.columns else target

    for col in hierarchy_cols:
        if col in df.columns:
            group_mean = df.groupby([DATE_COLUMN, col])[lag_col].transform("mean")
            df[f"{col}_avg_lag1"] = group_mean

    log.info("Added hierarchical features for: %s", hierarchy_cols)
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    target: str = PRIMARY_TARGET,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    include_hierarchical: bool = True,
) -> pd.DataFrame:
    """Run all feature engineering steps in order."""
    df = add_time_features(df)
    df = add_lag_features(df, target=target, lags=lags)
    df = add_rolling_features(df, target=target, windows=rolling_windows)
    df = add_comparative_features(df, target=target)
    if include_hierarchical:
        df = add_hierarchical_features(df, target=target)
    log.info("Feature engineering complete. Shape: %s", df.shape)
    return df
