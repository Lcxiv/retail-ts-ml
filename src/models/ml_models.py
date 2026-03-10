"""
Machine Learning Forecasting Models.

Uses a global pooled modeling strategy:
- Retailer and object are label-encoded as features
- Target: primary_target
- Validation: time-based (no shuffle)

Models: XGBoost, LightGBM
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)


def _encode_categoricals(df: pd.DataFrame, cat_cols: list[str]) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode categorical columns. Returns transformed df + encoders."""
    encoders: dict[str, LabelEncoder] = {}
    df = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def _time_split(df: pd.DataFrame, date_col: str, cutoff_frac: float = 0.8):
    """Split df into train/test based on time. No data shuffling."""
    dates = df[date_col].sort_values()
    cutoff_date = dates.quantile(cutoff_frac)
    train = df[df[date_col] <= cutoff_date]
    test = df[df[date_col] > cutoff_date]
    return train, test


def train_xgboost(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    date_col: str,
    cat_cols: list[str],
    cutoff_frac: float = 0.8,
    xgb_params: dict[str, Any] | None = None,
):
    """
    Train an XGBoost regressor on engineered features.

    Returns
    -------
    model : trained XGBRegressor
    test_df : test-set DataFrame with predictions column added
    encoders : label encoder dict
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("Install xgboost: pip install xgboost")

    df, encoders = _encode_categoricals(df, cat_cols)
    train, test = _time_split(df, date_col, cutoff_frac)

    X_train = train[feature_cols].fillna(0)
    y_train = train[target_col].fillna(0)
    X_test = test[feature_cols].fillna(0)

    params = xgb_params or {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, test[target_col].fillna(0))], verbose=False)

    test = test.copy()
    test["prediction"] = np.maximum(0, model.predict(X_test))
    log.info("XGBoost trained. Feature importances computed.")
    return model, test, encoders


def train_lightgbm(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    date_col: str,
    cat_cols: list[str],
    cutoff_frac: float = 0.8,
    lgb_params: dict[str, Any] | None = None,
):
    """
    Train a LightGBM regressor on engineered features.

    Returns
    -------
    model : trained LGBMRegressor
    test_df : test-set DataFrame with predictions column added
    encoders : label encoder dict
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("Install lightgbm: pip install lightgbm")

    df, encoders = _encode_categoricals(df, cat_cols)
    train, test = _time_split(df, date_col, cutoff_frac)

    X_train = train[feature_cols].fillna(0)
    y_train = train[target_col].fillna(0)
    X_test = test[feature_cols].fillna(0)

    params = lgb_params or {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)

    test = test.copy()
    test["prediction"] = np.maximum(0, model.predict(X_test))
    log.info("LightGBM trained.")
    return model, test, encoders


def feature_importance_df(model, feature_cols: list[str]) -> pd.DataFrame:
    """Return feature importance as a sorted DataFrame."""
    fi = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
    return fi.sort_values("importance", ascending=False).reset_index(drop=True)
