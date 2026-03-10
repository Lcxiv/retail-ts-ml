"""Tests for feature_engineering.py — feature creation and leakage prevention."""

import pytest
import pandas as pd
import numpy as np

from src.features.feature_engineering import (
    add_lag_features,
    add_rolling_features,
    add_time_features,
    add_comparative_features,
    add_hierarchical_features,
    build_features,
)
from src.config.config import DATE_COLUMN, RETAILER_COLUMN, STORE_COLUMN, PRODUCT_COLUMN


def _make_test_df(n_days=100) -> pd.DataFrame:
    """Create a small harmonized test DataFrame."""
    dates = pd.date_range("2022-01-01", periods=n_days, freq="W-MON")
    rows = []
    for store in [1, 2]:
        for product in ["UPC-A", "UPC-B"]:
            for d in dates:
                rows.append({
                    "date": d,
                    "retailer": "DG",
                    "store": store,
                    "product": product,
                    "qty_sold": max(0, np.random.normal(100, 20)),
                    "net_sales": max(0, np.random.normal(500, 100)),
                    "category": "Snacks",
                    "department": "Consumables",
                })
    return pd.DataFrame(rows)


class TestLagFeatures:
    def test_lag_columns_created(self):
        df = _make_test_df()
        result = add_lag_features(df, target="qty_sold", lags=[1, 2, 4])
        assert "lag_1" in result.columns
        assert "lag_2" in result.columns
        assert "lag_4" in result.columns

    def test_no_future_leakage(self):
        """First rows in each group should have NaN lags."""
        df = _make_test_df()
        result = add_lag_features(
            df, target="qty_sold", lags=[1],
            group_cols=[RETAILER_COLUMN, STORE_COLUMN, PRODUCT_COLUMN],
        )
        # First row of each group should be NaN for lag_1
        for (r, s, p), grp in result.groupby([RETAILER_COLUMN, STORE_COLUMN, PRODUCT_COLUMN]):
            assert pd.isna(grp["lag_1"].iloc[0]), f"Lag leakage in group ({r}, {s}, {p})"


class TestRollingFeatures:
    def test_rolling_columns_created(self):
        df = _make_test_df()
        result = add_rolling_features(df, target="qty_sold", windows=[4])
        assert "rolling_mean_4" in result.columns
        assert "rolling_std_4" in result.columns
        assert "rolling_max_4" in result.columns
        assert "rolling_min_4" in result.columns


class TestTimeFeatures:
    def test_calendar_features_added(self):
        df = _make_test_df(20)
        result = add_time_features(df)
        expected = ["year", "quarter", "month", "week", "day_of_year",
                    "week_index", "is_q4", "is_holiday_season"]
        for col in expected:
            assert col in result.columns, f"Missing time feature: {col}"

    def test_is_q4_flag(self):
        df = _make_test_df(60)
        result = add_time_features(df)
        q4_rows = result[result["quarter"] == 4]
        if len(q4_rows) > 0:
            assert (q4_rows["is_q4"] == 1).all()


class TestBuildFeatures:
    def test_full_pipeline_runs(self):
        """build_features should run without errors on valid data."""
        df = _make_test_df(60)
        result = build_features(df, target="qty_sold", lags=[1, 2], rolling_windows=[4])
        assert len(result) == len(df)
        assert "lag_1" in result.columns
        assert "rolling_mean_4" in result.columns
        assert "year" in result.columns

    def test_output_has_more_columns(self):
        df = _make_test_df(60)
        result = build_features(df, target="qty_sold")
        assert len(result.columns) > len(df.columns)
