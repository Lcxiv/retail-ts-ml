"""Tests for validation.py — data quality and distribution assessment."""

import pytest
import pandas as pd
import numpy as np

from src.data.validation import (
    check_stationarity,
    assess_distribution,
    detect_outliers,
    check_class_balance,
    recommend_model_family,
    generate_validation_report,
    ModelRecommendation,
)


class TestCheckStationarity:
    def test_stationary_series(self):
        """White noise should be stationary."""
        np.random.seed(42)
        s = pd.Series(np.random.normal(0, 1, 200))
        result = check_stationarity(s)
        if result["is_stationary"] is not None:  # statsmodels installed
            assert result["is_stationary"] is True
            assert result["p_value"] < 0.05

    def test_nonstationary_series(self):
        """Random walk should be non-stationary."""
        np.random.seed(42)
        s = pd.Series(np.cumsum(np.random.normal(0, 1, 200)))
        result = check_stationarity(s)
        if result["is_stationary"] is not None:
            assert result["is_stationary"] is False
            assert result["p_value"] >= 0.05

    def test_short_series_returns_none(self):
        s = pd.Series([1, 2, 3])
        result = check_stationarity(s)
        assert result["is_stationary"] is None

    def test_result_has_reasoning(self):
        np.random.seed(42)
        s = pd.Series(np.random.normal(0, 1, 100))
        result = check_stationarity(s)
        assert "reasoning" in result
        assert len(result["reasoning"]) > 0


class TestAssessDistribution:
    def test_normal_data(self):
        np.random.seed(42)
        s = pd.Series(np.random.normal(100, 10, 500))
        result = assess_distribution(s)
        assert abs(result["skewness"]) < 1.0
        assert result["mean"] is not None
        assert result["std"] is not None

    def test_skewed_data_recommends_transform(self):
        np.random.seed(42)
        s = pd.Series(np.random.exponential(10, 500))
        result = assess_distribution(s)
        assert result["skewness"] > 1.0
        assert result["recommended_transform"] == "log1p"

    def test_zero_inflated_data(self):
        np.random.seed(42)
        s = pd.Series([0] * 200 + list(np.random.exponential(10, 100)))
        result = assess_distribution(s)
        assert result["zero_fraction"] > 0.3
        assert result["recommended_transform"] == "two_stage_model"

    def test_result_has_reasoning(self):
        np.random.seed(42)
        s = pd.Series(np.random.normal(100, 10, 100))
        result = assess_distribution(s)
        assert "reasoning" in result


class TestDetectOutliers:
    def test_clean_data_few_outliers(self):
        np.random.seed(42)
        s = pd.Series(np.random.normal(0, 1, 1000))
        result = detect_outliers(s)
        assert result["outlier_fraction"] < 0.05

    def test_data_with_outliers(self):
        np.random.seed(42)
        s = pd.Series(list(np.random.normal(0, 1, 990)) + [100, -100] * 5)
        result = detect_outliers(s)
        assert result["n_outliers_combined"] > 0

    def test_recommendation_types(self):
        np.random.seed(42)
        s = pd.Series(np.random.normal(0, 1, 1000))
        result = detect_outliers(s)
        assert result["recommendation"] in ["none", "winsorize", "investigate"]


class TestCheckClassBalance:
    def test_balanced_groups(self):
        df = pd.DataFrame({
            "retailer": ["DG"] * 100 + ["SF"] * 100 + ["WG"] * 100,
            "qty_sold": np.random.randint(1, 100, 300),
        })
        result = check_class_balance(df, ["retailer"])
        assert result["is_balanced"] is True
        assert result["n_groups"] == 3

    def test_imbalanced_groups(self):
        df = pd.DataFrame({
            "retailer": ["DG"] * 1000 + ["SF"] * 10,
            "qty_sold": np.random.randint(1, 100, 1010),
        })
        result = check_class_balance(df, ["retailer"])
        assert result["is_balanced"] is False
        assert result["size_ratio"] > 5


class TestRecommendModelFamily:
    def test_returns_list_of_recommendations(self):
        stationarity = {"is_stationary": True, "p_value": 0.01}
        distribution = {"skewness": 0.5, "is_normal": True, "zero_fraction": 0.01}
        outliers = {"outlier_fraction": 0.01}
        balance = {"is_balanced": True}

        recs = recommend_model_family(
            stationarity, distribution, outliers, balance,
            n_series=10, n_points_per_series=200,
        )
        assert isinstance(recs, list)
        assert len(recs) > 0
        assert all(isinstance(r, ModelRecommendation) for r in recs)

    def test_sorted_by_suitability(self):
        stationarity = {"is_stationary": True, "p_value": 0.01}
        distribution = {"skewness": 0.5, "is_normal": True, "zero_fraction": 0.01}
        outliers = {"outlier_fraction": 0.01}
        balance = {"is_balanced": True}

        recs = recommend_model_family(
            stationarity, distribution, outliers, balance,
            n_series=10, n_points_per_series=200,
        )
        suit_order = {"high": 0, "medium": 1, "low": 2}
        for i in range(len(recs) - 1):
            assert suit_order[recs[i].suitability] <= suit_order[recs[i + 1].suitability]

    def test_lstm_low_with_small_data(self):
        stationarity = {"is_stationary": True, "p_value": 0.01}
        distribution = {"skewness": 0.5, "is_normal": True, "zero_fraction": 0.01}
        outliers = {"outlier_fraction": 0.01}
        balance = {"is_balanced": True}

        recs = recommend_model_family(
            stationarity, distribution, outliers, balance,
            n_series=3, n_points_per_series=50,
        )
        lstm_rec = [r for r in recs if "LSTM" in r.model_family][0]
        assert lstm_rec.suitability == "low"
