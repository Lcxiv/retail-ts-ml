"""Data Validation & Distribution Assessment Module.

Provides both automated analysis and advisory recommendations for:
- Stationarity testing (ADF test)
- Distribution shape analysis (skewness, kurtosis, normality)
- Outlier detection (IQR + Z-score)
- Class balance across groups
- Model family recommendation based on data characteristics

Design Philosophy
-----------------
This module serves a dual purpose:

1. **Automated**: Functions return structured results (dicts, DataFrames)
   that can be used programmatically in pipelines.
2. **Advisory**: `print_validation_report()` outputs human-readable
   recommendations explaining *why* each model family is suggested.

Usage:
    from src.data.validation import generate_validation_report, print_validation_report
    report = generate_validation_report(df, target_col="qty_sold", date_col="date")
    print_validation_report(report)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

log = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Stationarity Testing
# -------------------------------------------------------------------------

def check_stationarity(
    series: pd.Series,
    significance: float = 0.05,
) -> dict:
    """
    Augmented Dickey-Fuller test for stationarity.

    Parameters
    ----------
    series : pd.Series
        The time series to test (should be a single univariate series).
    significance : float
        p-value threshold for stationarity.

    Returns
    -------
    dict with keys:
        - is_stationary (bool)
        - adf_statistic (float)
        - p_value (float)
        - critical_values (dict)
        - reasoning (str): human-readable explanation

    ML Reasoning
    ------------
    - Stationary series: safe for ARIMA without differencing, linear models
    - Non-stationary series: need differencing (ARIMA d>0), or use tree-based
      models which handle non-stationarity through lag features
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return {
            "is_stationary": None,
            "adf_statistic": None,
            "p_value": None,
            "critical_values": {},
            "reasoning": "statsmodels not installed. Run: pip install statsmodels",
        }

    clean = series.dropna()
    if len(clean) < 20:
        return {
            "is_stationary": None,
            "adf_statistic": None,
            "p_value": None,
            "critical_values": {},
            "reasoning": f"Insufficient data ({len(clean)} points). Need at least 20.",
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = adfuller(clean, autolag="AIC")

    adf_stat, p_value, _, _, crit_values, _ = result
    is_stationary = p_value < significance

    if is_stationary:
        reasoning = (
            f"Series IS stationary (p={p_value:.4f} < {significance}). "
            f"ARIMA(p,0,q) is appropriate. Linear models and exponential smoothing "
            f"can be applied directly without differencing."
        )
    else:
        reasoning = (
            f"Series is NOT stationary (p={p_value:.4f} >= {significance}). "
            f"For ARIMA, use d>=1 (differencing). Prophet handles this automatically. "
            f"Tree-based models (XGBoost/LightGBM) are robust to non-stationarity "
            f"when using lag features as inputs."
        )

    return {
        "is_stationary": is_stationary,
        "adf_statistic": float(adf_stat),
        "p_value": float(p_value),
        "critical_values": {k: float(v) for k, v in crit_values.items()},
        "reasoning": reasoning,
    }


# -------------------------------------------------------------------------
# Distribution Analysis
# -------------------------------------------------------------------------

def assess_distribution(
    series: pd.Series,
    normality_alpha: float = 0.05,
) -> dict:
    """
    Analyze the distribution of a numeric series.

    Returns
    -------
    dict with keys:
        - mean, median, std (float)
        - skewness (float): 0=symmetric, >0=right-skewed, <0=left-skewed
        - kurtosis (float): 0=normal, >0=heavy-tailed, <0=light-tailed
        - is_normal (bool): Shapiro-Wilk test result
        - normality_p_value (float)
        - recommended_transform (str | None)
        - reasoning (str)

    ML Reasoning
    ------------
    - Normal distribution: linear regression, ARIMA work well with MSE loss
    - Right-skewed (common in sales): log-transform for linear models;
      tree-based models handle skew natively
    - Heavy-tailed: MAE/Huber loss preferred over MSE (robust to outliers)
    - Zero-inflated: consider two-stage model (classify zero/non-zero, then regress)
    """
    clean = series.dropna().astype(float)
    if len(clean) < 8:
        return {
            "mean": None, "median": None, "std": None,
            "skewness": None, "kurtosis": None,
            "is_normal": None, "normality_p_value": None,
            "zero_fraction": None,
            "recommended_transform": None,
            "reasoning": f"Insufficient data ({len(clean)} points).",
        }

    skew = float(sp_stats.skew(clean))
    kurt = float(sp_stats.kurtosis(clean))  # excess kurtosis (0 = normal)
    zero_frac = float((clean == 0).mean())

    # Shapiro-Wilk for normality (use subsample if >5000)
    test_sample = clean if len(clean) <= 5000 else clean.sample(5000, random_state=42)
    _, norm_p = sp_stats.shapiro(test_sample)
    is_normal = norm_p > normality_alpha

    # Determine transform recommendation
    transform = None
    reasoning_parts = []

    if zero_frac > 0.3:
        transform = "two_stage_model"
        reasoning_parts.append(
            f"{zero_frac:.1%} of values are zero → consider a two-stage model "
            f"(classify zero/non-zero first, then regress on non-zero values)."
        )
    elif abs(skew) > 1.0:
        if skew > 0:
            transform = "log1p"
            reasoning_parts.append(
                f"Right-skewed (skewness={skew:.2f}) → apply log(1+x) transform "
                f"for linear/ARIMA models. Tree-based models handle this naturally."
            )
        else:
            transform = "sqrt_reflect"
            reasoning_parts.append(
                f"Left-skewed (skewness={skew:.2f}) → reflect and sqrt transform, "
                f"or use tree-based models which are robust to skew."
            )
    elif not is_normal:
        transform = "none_but_monitor"
        reasoning_parts.append(
            f"Distribution is not normal (Shapiro p={norm_p:.4f}) but skewness is "
            f"moderate ({skew:.2f}). Linear models may still work; monitor residuals."
        )
    else:
        reasoning_parts.append(
            f"Distribution is approximately normal (Shapiro p={norm_p:.4f}). "
            f"All model families are appropriate."
        )

    if kurt > 3:
        reasoning_parts.append(
            f"Heavy-tailed (kurtosis={kurt:.2f}) → use MAE or Huber loss instead of "
            f"MSE to reduce sensitivity to extreme values."
        )

    return {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std()),
        "skewness": skew,
        "kurtosis": kurt,
        "is_normal": is_normal,
        "normality_p_value": float(norm_p),
        "zero_fraction": zero_frac,
        "recommended_transform": transform,
        "reasoning": " ".join(reasoning_parts),
    }


# -------------------------------------------------------------------------
# Outlier Detection
# -------------------------------------------------------------------------

def detect_outliers(
    series: pd.Series,
    method: str = "both",
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
) -> dict:
    """
    Detect outliers using IQR and/or Z-score methods.

    Parameters
    ----------
    method : 'iqr', 'zscore', or 'both'
    iqr_multiplier : IQR fence multiplier (1.5 = standard, 3.0 = extreme)
    zscore_threshold : Z-score cutoff for flagging

    Returns
    -------
    dict with:
        - n_outliers_iqr, n_outliers_zscore, n_outliers_combined (int)
        - outlier_fraction (float)
        - outlier_indices (np.ndarray)
        - recommendation (str): clip, winsorize, investigate, or none
        - reasoning (str)

    ML Reasoning
    ------------
    - <1% outliers: ignore or clip at boundaries
    - 1-5% outliers: winsorize (cap at percentile boundaries)
    - >5% outliers: investigate data quality; may be real signal (promos, events)
    - For RMSE-optimized models: outliers disproportionately affect loss
    - For tree models: outliers affect leaf predictions but are naturally isolated
    """
    clean = series.dropna().astype(float)
    n = len(clean)

    iqr_mask = np.zeros(n, dtype=bool)
    z_mask = np.zeros(n, dtype=bool)

    if method in ("iqr", "both"):
        q1, q3 = np.percentile(clean, [25, 75])
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        iqr_mask = (clean < lower) | (clean > upper)

    if method in ("zscore", "both"):
        z = np.abs(sp_stats.zscore(clean))
        z_mask = z > zscore_threshold

    combined = iqr_mask | z_mask
    frac = float(combined.mean())

    if frac < 0.01:
        recommendation = "none"
        reasoning = f"Only {frac:.1%} outliers detected — negligible impact. No treatment needed."
    elif frac < 0.05:
        recommendation = "winsorize"
        reasoning = (
            f"{frac:.1%} outliers detected. Recommend winsorizing (capping at 1st/99th "
            f"percentile) for RMSE-sensitive models. Tree models are less affected."
        )
    else:
        recommendation = "investigate"
        reasoning = (
            f"{frac:.1%} outliers detected — unusually high. Investigate whether these "
            f"represent real events (promotions, stockouts) or data quality issues. "
            f"If real, consider adding event features instead of removing."
        )

    return {
        "n_outliers_iqr": int(iqr_mask.sum()),
        "n_outliers_zscore": int(z_mask.sum()),
        "n_outliers_combined": int(combined.sum()),
        "outlier_fraction": frac,
        "outlier_indices": np.where(combined)[0],
        "recommendation": recommendation,
        "reasoning": reasoning,
    }


# -------------------------------------------------------------------------
# Class / Group Balance
# -------------------------------------------------------------------------

def check_class_balance(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str = "qty_sold",
) -> dict:
    """
    Check distribution of data across groups (retailers, stores, etc.).

    Returns
    -------
    dict with:
        - n_groups (int)
        - group_sizes (pd.Series)
        - size_ratio (float): max_group / min_group
        - is_balanced (bool): ratio < 5
        - reasoning (str)

    ML Reasoning
    ------------
    - Balanced groups: global pooled model is appropriate
    - Imbalanced groups (>5x): consider per-group models, or weighted sampling
    - Extreme imbalance (>20x): global model will be dominated by majority group
    """
    sizes = df.groupby(group_cols).size()
    ratio = float(sizes.max() / max(sizes.min(), 1))
    is_balanced = ratio < 5

    if is_balanced:
        reasoning = (
            f"{len(sizes)} groups with size ratio {ratio:.1f}x. "
            f"Balanced → global pooled model is appropriate."
        )
    elif ratio < 20:
        reasoning = (
            f"{len(sizes)} groups with size ratio {ratio:.1f}x (moderately imbalanced). "
            f"Consider sample weighting or stratified train/test splits. "
            f"Alternatively, train separate models per dominant group."
        )
    else:
        reasoning = (
            f"{len(sizes)} groups with size ratio {ratio:.1f}x (severely imbalanced). "
            f"A global model will be dominated by the largest group. "
            f"Train separate per-group models or use hierarchical modeling."
        )

    return {
        "n_groups": len(sizes),
        "group_sizes": sizes,
        "size_ratio": ratio,
        "is_balanced": is_balanced,
        "reasoning": reasoning,
    }


# -------------------------------------------------------------------------
# Model Family Recommendation
# -------------------------------------------------------------------------

@dataclass
class ModelRecommendation:
    """A single model recommendation with confidence and reasoning."""
    model_family: str           # e.g., "XGBoost", "ARIMA", "Prophet"
    suitability: str            # "high", "medium", "low"
    reasoning: str              # why this model fits (or doesn't)
    prerequisites: list[str] = field(default_factory=list)  # what to do first


def recommend_model_family(
    stationarity: dict,
    distribution: dict,
    outliers: dict,
    balance: dict,
    n_series: int = 1,
    n_points_per_series: int = 100,
) -> list[ModelRecommendation]:
    """
    Recommend model families based on data validation results.

    This function is BOTH automated (returns structured recommendations)
    AND advisory (each recommendation includes reasoning).

    Parameters
    ----------
    stationarity : dict from check_stationarity()
    distribution : dict from assess_distribution()
    outliers : dict from detect_outliers()
    balance : dict from check_class_balance()
    n_series : int - number of distinct time series
    n_points_per_series : int - average data points per series

    Returns
    -------
    list[ModelRecommendation] sorted by suitability (high → low)
    """
    recommendations = []
    is_stationary = stationarity.get("is_stationary", True)
    skew = abs(distribution.get("skewness", 0) or 0)
    is_normal = distribution.get("is_normal", True)
    zero_frac = distribution.get("zero_fraction", 0) or 0
    outlier_frac = outliers.get("outlier_fraction", 0) or 0
    is_balanced = balance.get("is_balanced", True)

    # --- XGBoost / LightGBM ---
    xgb_suit = "high"
    xgb_reasons = []
    xgb_prereqs = []
    if n_series > 1:
        xgb_reasons.append("Excels at multi-series forecasting with cross-learning.")
    if skew > 1.0:
        xgb_reasons.append("Handles skewed distributions natively (no transform needed).")
    if not is_stationary:
        xgb_reasons.append("Robust to non-stationarity through lag features.")
    if outlier_frac > 0.03:
        xgb_reasons.append("Isolates outliers in tree leaves; less affected than linear models.")
    xgb_reasons.append("Works well with engineered tabular features (lags, rolling stats).")
    if n_points_per_series < 50:
        xgb_suit = "medium"
        xgb_reasons.append("Limited data per series — risk of overfitting.")
        xgb_prereqs.append("Use regularization: max_depth=4, min_child_weight=10.")

    recommendations.append(ModelRecommendation(
        model_family="XGBoost / LightGBM",
        suitability=xgb_suit,
        reasoning=" ".join(xgb_reasons),
        prerequisites=xgb_prereqs,
    ))

    # --- ARIMA / SARIMAX ---
    arima_suit = "medium"
    arima_reasons = []
    arima_prereqs = []
    if is_stationary:
        arima_suit = "high" if n_series == 1 else "medium"
        arima_reasons.append("Stationary data — ARIMA(p,0,q) is directly applicable.")
    else:
        arima_reasons.append("Non-stationary — requires differencing (d>=1).")
        arima_prereqs.append("Apply differencing or use auto_arima for automatic order selection.")
    if n_series > 10:
        arima_suit = "low"
        arima_reasons.append(
            f"With {n_series} series, fitting individual ARIMA models is slow "
            f"and doesn't allow cross-learning between series."
        )
    if not is_normal and skew > 1.0:
        arima_prereqs.append("Apply log-transform before fitting (ARIMA assumes ~normal residuals).")
        arima_reasons.append("Skewed data — needs transformation for valid inference.")
    arima_reasons.append("Best for interpretable, single-series analysis.")

    recommendations.append(ModelRecommendation(
        model_family="ARIMA / SARIMAX",
        suitability=arima_suit,
        reasoning=" ".join(arima_reasons),
        prerequisites=arima_prereqs,
    ))

    # --- Prophet ---
    prophet_suit = "medium"
    prophet_reasons = []
    prophet_prereqs = []
    prophet_reasons.append("Handles seasonality, trend changes, and holidays automatically.")
    if not is_stationary:
        prophet_reasons.append("Handles non-stationarity well (piecewise linear trend).")
        prophet_suit = "high" if n_series <= 10 else "medium"
    if n_series > 50:
        prophet_suit = "low"
        prophet_reasons.append(
            f"Fitting {n_series} individual Prophet models is computationally expensive."
        )
    if zero_frac > 0.2:
        prophet_reasons.append("Struggles with zero-inflated data.")
        prophet_suit = "low"
    prophet_reasons.append("Good baseline for series with strong seasonal patterns.")

    recommendations.append(ModelRecommendation(
        model_family="Prophet",
        suitability=prophet_suit,
        reasoning=" ".join(prophet_reasons),
        prerequisites=prophet_prereqs,
    ))

    # --- LSTM / Neural Networks ---
    nn_suit = "low"
    nn_reasons = []
    nn_prereqs = []
    if n_points_per_series >= 500 and n_series >= 50:
        nn_suit = "medium"
        nn_reasons.append(
            f"Sufficient data ({n_points_per_series} points × {n_series} series) "
            f"for neural network training."
        )
    else:
        nn_reasons.append(
            f"Insufficient data ({n_points_per_series} points × {n_series} series). "
            f"Neural networks typically need 1000+ data points per series to outperform GBMs."
        )
    nn_reasons.append(
        "LSTMs can capture complex temporal dependencies but require careful tuning "
        "(learning rate, sequence length, architecture)."
    )
    nn_prereqs.append("Normalize/standardize all features.")
    nn_prereqs.append("Implement proper sequence windowing without data leakage.")

    recommendations.append(ModelRecommendation(
        model_family="LSTM / Neural Network",
        suitability=nn_suit,
        reasoning=" ".join(nn_reasons),
        prerequisites=nn_prereqs,
    ))

    # --- Exponential Smoothing ---
    ets_suit = "medium"
    ets_reasons = ["Simple, fast, and interpretable."]
    ets_prereqs = []
    if n_series > 20:
        ets_suit = "low"
        ets_reasons.append("No cross-learning; must fit per series.")
    if is_stationary:
        ets_reasons.append("Works well with stationary data.")
    ets_reasons.append("Good as a baseline to benchmark against ML models.")

    recommendations.append(ModelRecommendation(
        model_family="Exponential Smoothing (ETS)",
        suitability=ets_suit,
        reasoning=" ".join(ets_reasons),
        prerequisites=ets_prereqs,
    ))

    # Sort: high > medium > low
    order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda r: order.get(r.suitability, 1))

    return recommendations


# -------------------------------------------------------------------------
# Comprehensive Report
# -------------------------------------------------------------------------

def generate_validation_report(
    df: pd.DataFrame,
    target_col: str = "qty_sold",
    date_col: str = "date",
    group_cols: list[str] | None = None,
) -> dict:
    """
    Run all validation checks and return a comprehensive report.

    Parameters
    ----------
    df : Harmonized DataFrame
    target_col : primary target
    date_col : date column
    group_cols : columns for balance checking (default: ["retailer"])

    Returns
    -------
    dict with keys: stationarity, distribution, outliers, balance, recommendations
    """
    if group_cols is None:
        group_cols = ["retailer"]

    # Aggregate to a single series for stationarity test
    agg_series = df.groupby(date_col)[target_col].sum().sort_index()

    log.info("Running stationarity test...")
    stationarity = check_stationarity(agg_series)

    log.info("Assessing target distribution...")
    distribution = assess_distribution(df[target_col])

    log.info("Detecting outliers...")
    outliers = detect_outliers(df[target_col])

    log.info("Checking class balance...")
    balance = check_class_balance(df, group_cols, target_col)

    # Count series and points
    valid_group_cols = [c for c in group_cols if c in df.columns]
    if valid_group_cols:
        series_counts = df.groupby(valid_group_cols).size()
        n_series = len(series_counts)
        n_points = int(series_counts.mean())
    else:
        n_series = 1
        n_points = len(df)

    log.info("Generating model recommendations...")
    recommendations = recommend_model_family(
        stationarity, distribution, outliers, balance,
        n_series=n_series,
        n_points_per_series=n_points,
    )

    return {
        "stationarity": stationarity,
        "distribution": distribution,
        "outliers": outliers,
        "balance": balance,
        "recommendations": recommendations,
        "data_summary": {
            "n_rows": len(df),
            "n_series": n_series,
            "avg_points_per_series": n_points,
            "date_range": (str(df[date_col].min()), str(df[date_col].max())),
        },
    }


def print_validation_report(report: dict) -> None:
    """
    Print a human-readable validation report with recommendations.
    This is the advisory mode — designed for interactive use.
    """
    print("=" * 70)
    print("          DATA VALIDATION & MODEL RECOMMENDATION REPORT")
    print("=" * 70)

    # Data summary
    s = report["data_summary"]
    print(f"\n📊 Data: {s['n_rows']:,} rows | {s['n_series']} series | "
          f"~{s['avg_points_per_series']} points/series")
    print(f"   Date range: {s['date_range'][0]} → {s['date_range'][1]}")

    # Stationarity
    st = report["stationarity"]
    icon = "✅" if st["is_stationary"] else "⚠️"
    print(f"\n{icon} STATIONARITY")
    print(f"   {st['reasoning']}")

    # Distribution
    d = report["distribution"]
    print(f"\n📈 DISTRIBUTION")
    print(f"   Mean={d['mean']:.2f} | Median={d['median']:.2f} | Std={d['std']:.2f}")
    print(f"   Skewness={d['skewness']:.2f} | Kurtosis={d['kurtosis']:.2f}")
    print(f"   Normal: {'Yes' if d['is_normal'] else 'No'} (p={d['normality_p_value']:.4f})")
    if d["recommended_transform"]:
        print(f"   Transform: {d['recommended_transform']}")
    print(f"   {d['reasoning']}")

    # Outliers
    o = report["outliers"]
    print(f"\n🔍 OUTLIERS")
    print(f"   {o['n_outliers_combined']} detected ({o['outlier_fraction']:.1%})")
    print(f"   Recommendation: {o['recommendation']}")
    print(f"   {o['reasoning']}")

    # Balance
    b = report["balance"]
    print(f"\n⚖️  GROUP BALANCE")
    print(f"   {b['reasoning']}")

    # Recommendations
    print(f"\n{'=' * 70}")
    print("          MODEL RECOMMENDATIONS (sorted by suitability)")
    print("=" * 70)
    for rec in report["recommendations"]:
        icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(rec.suitability, "⚪")
        print(f"\n{icon} {rec.model_family} [{rec.suitability.upper()}]")
        print(f"   {rec.reasoning}")
        if rec.prerequisites:
            print(f"   Prerequisites:")
            for p in rec.prerequisites:
                print(f"     • {p}")

    print(f"\n{'=' * 70}")
