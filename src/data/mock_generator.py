"""
Synthetic Retail Sales Data Generator.

Generates realistic mock time series data for 4 retailers and 6 selling objects.
All assumptions are documented inline. Replace with real data once schema is confirmed.

Usage:
    python -m src.data.mock_generator
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.config import (
    RETAILER_PROFILES,
    OBJECT_PROFILES,
    SYNTH_START_DATE,
    SYNTH_END_DATE,
    FREQUENCY_ALIAS,
    DATA_SYNTHETIC_DIR,
    PRIMARY_TARGET,
    DATE_COLUMN,
    RETAILER_COLUMN,
    OBJECT_COLUMN,
    CATEGORY_COLUMN,
    BRAND_COLUMN,
    REGION_COLUMN,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

NP_SEED = 42
rng = np.random.default_rng(NP_SEED)
random.seed(NP_SEED)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _make_time_index() -> pd.DatetimeIndex:
    """Build weekly date range for the synthetic period."""
    return pd.date_range(start=SYNTH_START_DATE, end=SYNTH_END_DATE, freq=FREQUENCY_ALIAS)


def _trend_component(n: int, slope: float) -> np.ndarray:
    """Linear trend growing at `slope` units per week."""
    return slope * np.arange(n)


def _seasonality_component(
    dates: pd.DatetimeIndex, strength: float, peak_season: str | None = None
) -> np.ndarray:
    """
    Seasonality signal:
    - Default: double peak (holiday Q4 + mid-year bump)
    - summer peak_season: shifts main peak to weeks 24-35
    """
    week = dates.isocalendar().week.to_numpy(dtype=float)
    if peak_season == "summer":
        # Summer peak around week 28
        signal = strength * 60 * np.exp(-0.5 * ((week - 28) / 6) ** 2)
    else:
        # Holiday peak (week 48) + mild summer bump (week 26)
        signal = (
            strength * 80 * np.exp(-0.5 * ((week - 48) / 5) ** 2)
            + strength * 30 * np.exp(-0.5 * ((week - 26) / 6) ** 2)
        )
    return signal


def _noise_component(n: int, level: str, base: float) -> np.ndarray:
    """Add multiplicative noise scaled to base demand."""
    scale_map = {"low": 0.03, "medium": 0.08, "high": 0.18}
    sigma = scale_map.get(level, 0.08) * base
    return rng.normal(0, sigma, n)


def _promo_spikes(n: int, freq: str, base: float) -> np.ndarray:
    """Inject random promotional spikes."""
    spikes = np.zeros(n)
    rate_map = {"high": 0.12, "medium": 0.06, "low": 0.02}
    rate = rate_map.get(freq, 0.04)
    spike_idx = rng.choice(n, size=int(n * rate), replace=False)
    spikes[spike_idx] = rng.uniform(0.5 * base, 1.5 * base, len(spike_idx))
    return spikes


def _anomalies(n: int, base: float) -> np.ndarray:
    """Inject rare positive/negative anomalies."""
    anomaly = np.zeros(n)
    n_events = max(1, int(n * 0.015))
    idx = rng.choice(n, size=n_events, replace=False)
    magnitudes = rng.choice([-1, 1], size=n_events) * rng.uniform(0.4 * base, 0.9 * base, n_events)
    anomaly[idx] = magnitudes
    return anomaly


def _inject_missingness(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Simulate realistic missingness:
    - ~2% random missing dates
    - cold-start effect: first 8 weeks of some objects have NaN
    """
    df = df.copy()
    n = len(df)
    # Random missing
    missing_idx = rng.choice(n, size=int(n * 0.02), replace=False)
    df.loc[df.index[missing_idx], target_col] = np.nan
    return df


def _generate_series(
    dates: pd.DatetimeIndex,
    retailer: str,
    obj_id: str,
    obj_profile: dict,
    ret_profile: dict,
) -> pd.DataFrame:
    """Generate a single (retailer, object) time series."""
    n = len(dates)
    base = obj_profile["base_units"] * (ret_profile["base_units"] / 400)

    trend = _trend_component(n, ret_profile["trend_slope"])
    season = _seasonality_component(
        dates,
        ret_profile["seasonality_strength"],
        ret_profile.get("peak_season"),
    )
    noise = _noise_component(n, ret_profile["noise_level"], base)
    promo = _promo_spikes(n, ret_profile.get("promo_frequency", "low"), base)
    anomaly = _anomalies(n, base)

    units = np.maximum(0, base + trend + season + noise + promo + anomaly)

    # Derived financials (assumptions documented)
    price = rng.uniform(10, 50, n)  # placeholder price
    margin_pct = rng.uniform(0.15, 0.40, n)  # 15-40% margin assumption
    revenue = units * price
    margin = revenue * margin_pct

    promo_flag = (promo > 0).astype(int)
    holiday_flag = (dates.month == 12).astype(int)  # simple Dec holiday flag

    df = pd.DataFrame(
        {
            DATE_COLUMN: dates,
            RETAILER_COLUMN: retailer,
            OBJECT_COLUMN: obj_id,
            CATEGORY_COLUMN: obj_profile["category"],
            BRAND_COLUMN: obj_profile["brand"],
            REGION_COLUMN: obj_profile["region"],
            PRIMARY_TARGET: units,
            "revenue": revenue,
            "margin": margin,
            "price": price,
            "promo_flag": promo_flag,
            "holiday_flag": holiday_flag,
        }
    )
    df = _inject_missingness(df, PRIMARY_TARGET)
    return df


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_synthetic_data(save: bool = True) -> pd.DataFrame:
    """
    Generate synthetic retail sales dataset.

    Returns
    -------
    pd.DataFrame
        Long-format dataframe with one row per (date, retailer, object).
    """
    dates = _make_time_index()
    log.info("Generating synthetic data: %d weeks (%s to %s)", len(dates), dates[0].date(), dates[-1].date())

    records: list[pd.DataFrame] = []
    for retailer, ret_profile in RETAILER_PROFILES.items():
        for obj_id, obj_profile in OBJECT_PROFILES.items():
            series_df = _generate_series(dates, retailer, obj_id, obj_profile, ret_profile)
            records.append(series_df)

    df = pd.concat(records, ignore_index=True)
    df = df.sort_values([DATE_COLUMN, RETAILER_COLUMN, OBJECT_COLUMN]).reset_index(drop=True)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    log.info("Dataset shape: %s", df.shape)
    log.info("Retailers: %s", df[RETAILER_COLUMN].unique().tolist())
    log.info("Objects: %s", df[OBJECT_COLUMN].unique().tolist())
    log.info("Date range: %s to %s", df[DATE_COLUMN].min().date(), df[DATE_COLUMN].max().date())
    log.info("Missing %s: %.2f%%", PRIMARY_TARGET, df[PRIMARY_TARGET].isna().mean() * 100)

    if save:
        out_path = DATA_SYNTHETIC_DIR / "synthetic_retail_sales.csv"
        df.to_csv(out_path, index=False)
        parquet_path = DATA_SYNTHETIC_DIR / "synthetic_retail_sales.parquet"
        df.to_parquet(parquet_path, index=False)
        log.info("Saved to %s", out_path)
        log.info("Saved to %s", parquet_path)

    return df


if __name__ == "__main__":
    generate_synthetic_data(save=True)
