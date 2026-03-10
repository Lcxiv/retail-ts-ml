"""
General-purpose utilities.
"""

from __future__ import annotations

import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from functools import wraps


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a clean format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------

def week_start(dt: pd.Timestamp, weekday: int = 0) -> pd.Timestamp:
    """Return the Monday (or given weekday) of the week containing `dt`."""
    days_behind = (dt.weekday() - weekday) % 7
    return dt - pd.Timedelta(days=days_behind)


def date_range_weeks(start: str, end: str) -> pd.DatetimeIndex:
    """Weekly date range starting on Monday."""
    return pd.date_range(start, end, freq="W-MON")


def time_periods_between(start: pd.Timestamp, end: pd.Timestamp, freq: str = "W") -> int:
    """Number of periods between two dates."""
    return len(pd.date_range(start, end, freq=freq)) - 1


# ---------------------------------------------------------------------------
# Safe math
# ---------------------------------------------------------------------------

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns `default` instead of inf/nan on zero denominator."""
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator


# ---------------------------------------------------------------------------
# DataFrame utilities
# ---------------------------------------------------------------------------

def fill_missing_dates(
    df: pd.DataFrame,
    date_col: str,
    entity_cols: list[str],
    freq: str = "W-MON",
) -> pd.DataFrame:
    """
    Ensure every (entity, date) combination is present in the dataframe.
    Missing rows are filled with NaN. Useful before lag/rolling computations.
    """
    all_dates = pd.date_range(df[date_col].min(), df[date_col].max(), freq=freq)
    entities = df[entity_cols].drop_duplicates()
    template = entities.merge(pd.DataFrame({date_col: all_dates}), how="cross")
    merged = template.merge(df, on=entity_cols + [date_col], how="left")
    return merged.sort_values(entity_cols + [date_col]).reset_index(drop=True)


def summarize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Quick profile: dtype, null count, null%, nunique, min, max."""
    rows = []
    for col in df.columns:
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "null_count": df[col].isna().sum(),
            "null_pct": round(df[col].isna().mean() * 100, 2),
            "nunique": df[col].nunique(),
            "min": df[col].min() if df[col].dtype != object else None,
            "max": df[col].max() if df[col].dtype != object else None,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------

def timed(fn):
    """Decorator that logs execution time of a function."""
    log = logging.getLogger(fn.__module__)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        log.info("%s completed in %.2fs", fn.__name__, elapsed)
        return result

    return wrapper
