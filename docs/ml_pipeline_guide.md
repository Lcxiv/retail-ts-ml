# 📊 Machine Learning Pipeline Guide

A comprehensive, reproducible step-by-step guide for the retail time series forecasting pipeline. Each step includes **what** to do, **runnable code**, and **why** each decision is made.

> **Target audience**: Data scientists reproducing this analysis, or reviewers evaluating methodology.

---

## Table of Contents

1. [Data Ingestion & Harmonization](#step-1-data-ingestion--harmonization)
2. [Exploratory Data Analysis](#step-2-exploratory-data-analysis)
3. [Data Validation & Distribution Assessment](#step-3-data-validation--distribution-assessment)
4. [Feature Engineering](#step-4-feature-engineering)
5. [Train/Test Splitting Strategy](#step-5-traintest-splitting-strategy)
6. [Baseline Models](#step-6-baseline-models)
7. [Advanced Model Selection & Reasoning](#step-7-advanced-model-selection--reasoning)
8. [Hyperparameter Tuning](#step-8-hyperparameter-tuning)
9. [Backtesting & Evaluation](#step-9-backtesting--evaluation)
10. [Model Comparison & Final Selection](#step-10-model-comparison--final-selection)

---

## Step 1: Data Ingestion & Harmonization

### What

Load raw data from each of the three database schemas (DG, S&F, WG) and convert them into a single unified format.

### Why

Each database has different column names, date formats, and granularity levels. Without harmonization, you'd need separate pipelines for each. The unified schema ensures:
- **Consistent feature engineering** regardless of source
- **Cross-retailer comparisons** using the same column names
- **Single model training** across all retailers (pooled model)

### Code

```python
import pandas as pd
from src.data.schema import DG_SCHEMA, SF_SCHEMA, WG_SCHEMA, harmonize_to_unified
from src.data.mock_generator import generate_all

# Option A: Generate synthetic data
datasets = generate_all(save=True)

# Option B: Load from existing files
dg_raw = pd.read_parquet("data/synthetic/synthetic_dg.parquet")
sf_raw = pd.read_parquet("data/synthetic/synthetic_sf.parquet")
wg_raw = pd.read_parquet("data/synthetic/synthetic_wg.parquet")

# Harmonize each database to unified schema
dg = harmonize_to_unified(dg_raw, DG_SCHEMA)
sf = harmonize_to_unified(sf_raw, SF_SCHEMA)
wg = harmonize_to_unified(wg_raw, WG_SCHEMA)

# Combine into a single DataFrame
df = pd.concat([dg, sf, wg], ignore_index=True)
print(f"Combined: {len(df):,} rows | Columns: {list(df.columns)}")
```

### Key Decisions

| Decision | Choice | Reasoning |
|---|---|---|
| Primary product key | `UPC` (first in each schema's product_keys) | Most granular; SKU is retailer-internal |
| Date format | `datetime64` | Required for time-based splitting and lag computation |
| S&F integer dates | Convert via `YYYYMMDD` → `datetime` | `TRANSACTIONDATE` is `NUMBER(38,0)` in Snowflake |
| Missing hierarchy | Fill with `"unknown"` | WG has no department; S&F has no category/department |

---

## Step 2: Exploratory Data Analysis

### What

Understand the data structure, identify patterns, and spot potential issues before modeling.

### Why

EDA informs **every downstream decision**: feature selection, model choice, evaluation strategy. Skipping EDA leads to models that learn noise, miss seasonality, or fail on edge cases.

### Code

```python
from src.visualization.plots import (
    plot_sales_trend, plot_seasonality_heatmap, plot_distribution,
    plot_rolling_average_comparison,
)
from src.utils.helpers import summarize_df

# Quick data profile
profile = summarize_df(df)
print(profile.to_string())

# 1. Sales trends by retailer
fig = plot_sales_trend(df, target_col="qty_sold", color_col="retailer")
fig.show()

# 2. Seasonality heatmap
fig = plot_seasonality_heatmap(df, target_col="qty_sold")
fig.show()

# 3. Target distribution
fig = plot_distribution(df, target_col="qty_sold", group_col="retailer")
fig.show()

# 4. Rolling average comparison
fig = plot_rolling_average_comparison(df, target_col="qty_sold", window=8)
fig.show()

# 5. Missing data check
for col in df.columns:
    null_pct = df[col].isna().mean() * 100
    if null_pct > 0:
        print(f"  {col}: {null_pct:.1f}% missing")
```

### What to Look For

| Pattern | Implication | Action |
|---|---|---|
| Clear upward/downward trend | Non-stationary → differencing for ARIMA | Add trend features, or use models that handle trends |
| Repeated seasonal spikes | Strong seasonality | Add Fourier terms or calendar features |
| Heavy right skew in target | MSE loss penalizes large errors disproportionately | Log-transform target, or use MAE loss |
| Many zeros | Zero-inflated distribution | Consider two-stage modeling |
| Retailer-specific patterns | Different dynamics per retailer | Per-retailer models may outperform global |

---

## Step 3: Data Validation & Distribution Assessment

### What

Formally test the statistical properties of the data to determine which model families are appropriate.

### Why

**"Choosing the right model for the right distribution"** is foundational in ML. A model optimized for normally distributed residuals will perform poorly on heavily skewed data. This step provides evidence-based model selection instead of guessing.

### Code

```python
from src.data.validation import (
    generate_validation_report,
    print_validation_report,
    check_stationarity,
    assess_distribution,
    detect_outliers,
    recommend_model_family,
)

# One-call comprehensive report (automated + advisory)
report = generate_validation_report(
    df,
    target_col="qty_sold",
    date_col="date",
    group_cols=["retailer"],
)

# Print human-readable recommendations
print_validation_report(report)

# Access individual results programmatically
print(f"Stationary: {report['stationarity']['is_stationary']}")
print(f"Skewness: {report['distribution']['skewness']:.2f}")
print(f"Outlier fraction: {report['outliers']['outlier_fraction']:.1%}")
print(f"Top model: {report['recommendations'][0].model_family}")
```

### Distribution → Model Selection Matrix

This is the core logic that maps data characteristics to model choices:

| Data Characteristic | ARIMA | Prophet | XGBoost/LightGBM | LSTM | ETS |
|---|---|---|---|---|---|
| **Stationary** | ✅ Direct fit | ✅ | ✅ | ✅ | ✅ |
| **Non-stationary** | ⚠️ Needs d≥1 | ✅ Auto-handles | ✅ Via lag features | ✅ | ⚠️ Limited |
| **Normal distribution** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Skewed distribution** | ⚠️ Log-transform first | ✅ | ✅ Native handling | ⚠️ Normalize | ⚠️ Transform |
| **Heavy outliers** | ❌ Sensitive | ✅ Robust | ✅ Tree isolation | ❌ Sensitive | ❌ Sensitive |
| **Many series (100+)** | ❌ Slow (per-series) | ❌ Slow | ✅ Cross-learning | ✅ Cross-learning | ❌ Per-series |
| **Few data points (<50/series)** | ⚠️ Low power | ✅ Works well | ⚠️ Overfitting risk | ❌ Needs 500+ | ✅ |
| **Complex interactions** | ❌ Linear only | ❌ Additive | ✅ Non-linear | ✅ Non-linear | ❌ |

### Per-Retailer Validation

```python
# Run validation per retailer to see if different models suit different data
for retailer in df["retailer"].unique():
    r_df = df[df["retailer"] == retailer]
    r_report = generate_validation_report(
        r_df, target_col="qty_sold", date_col="date", group_cols=["store"]
    )
    top = r_report["recommendations"][0]
    print(f"{retailer}: {top.model_family} [{top.suitability}] — {top.reasoning[:80]}...")
```

---

## Step 4: Feature Engineering

### What

Create informative features from raw data that help models learn temporal patterns.

### Why

Tree-based models (XGBoost, LightGBM) **cannot extrapolate** — they only learn from patterns seen in training data. Without explicit lag and calendar features, a GBM has no way to know "this week is similar to the same week last year." Features encode this temporal knowledge.

### Code

```python
from src.features.feature_engineering import build_features

# Run the full feature pipeline
df_features = build_features(
    df,
    target="qty_sold",
    lags=[1, 2, 4, 8, 12, 26, 52],
    rolling_windows=[4, 8, 12],
    include_hierarchical=True,
)

# Inspect new columns
new_cols = [c for c in df_features.columns if c not in df.columns]
print(f"Added {len(new_cols)} features: {new_cols}")

# Check for data leakage: first rows should have NaN lags
print("\nFirst row lag_1:", df_features["lag_1"].iloc[0])  # Should be NaN
print("First row lag_52:", df_features["lag_52"].iloc[0])  # Should be NaN
```

### Feature Groups & Reasoning

| Feature Group | Examples | Why It Helps |
|---|---|---|
| **Lag features** | `lag_1`, `lag_52` | Captures autocorrelation; "sales last week predict this week" |
| **Rolling statistics** | `rolling_mean_4`, `rolling_std_8` | Captures local trend and volatility |
| **Calendar features** | `month`, `week`, `is_holiday_season` | Captures seasonal patterns that repeat yearly |
| **Comparative features** | `store_share_of_retailer`, `store_zscore` | Normalizes across different scales |
| **Hierarchical features** | `category_avg_lag1` | Category-level signals help product-level forecasting |

### ⚠️ Data Leakage Prevention

All features use **shifted** (lagged) values:
- `rolling_mean_4` uses `.shift(1).rolling(4)` — excludes current period
- `lag_1` is `groupby(...).shift(1)` — previous period only
- No feature ever uses data from time `t` or later to predict `t`

---

## Step 5: Train/Test Splitting Strategy

### What

Split data into training and test sets using **time-based** splitting.

### Why

> **❌ NEVER use random train/test splits for time series.**
>
> Random splits cause **data leakage**: future observations end up in the training set, giving the model information it wouldn't have in production. This leads to overly optimistic metrics that don't reflect real forecasting performance.

### Code

```python
from src.config.config import TRAIN_CUTOFF_FRAC, DATE_COLUMN

# Sort by date
df_features = df_features.sort_values(DATE_COLUMN)

# Time-based split at 80th percentile of dates
dates = df_features[DATE_COLUMN].sort_values()
cutoff_date = dates.quantile(TRAIN_CUTOFF_FRAC)

train = df_features[df_features[DATE_COLUMN] <= cutoff_date]
test = df_features[df_features[DATE_COLUMN] > cutoff_date]

print(f"Train: {len(train):,} rows | {train[DATE_COLUMN].min()} → {train[DATE_COLUMN].max()}")
print(f"Test:  {len(test):,} rows  | {test[DATE_COLUMN].min()} → {test[DATE_COLUMN].max()}")
print(f"Cutoff: {cutoff_date}")
```

### Splitting Strategies Compared

| Strategy | Method | Pros | Cons | Use When |
|---|---|---|---|---|
| **Simple time split** | 80/20 by date | Fast, simple | Single evaluation point | Quick experiments |
| **Rolling origin CV** | Multiple expanding windows | Robust, realistic | Slower | Final evaluation |
| **Expanding window** | Train grows, test fixed size | Tests stability over time | Computationally expensive | Production validation |
| **❌ Random split** | `train_test_split(shuffle=True)` | ❌ Never for time series | Data leakage | **Never** |

---

## Step 6: Baseline Models

### What

Establish performance benchmarks using simple, interpretable models.

### Why

> **Any ML model that can’t beat a naive baseline has negative value.**

Baselines set the **performance floor**. If XGBoost only improves MAE by 1% over seasonal naive, the complexity isn't justified. Common practice: a model must beat the best baseline by at least 5-10% to be worth deploying.

### Code

```python
from src.models.baseline_models import run_all_baselines, ALL_BASELINES
from src.evaluation.metrics import all_metrics
import pandas as pd

# Select a single series for baseline testing
series_mask = (
    (train["retailer"] == "DG") & 
    (train["store"] == train["store"].iloc[0])
)
train_series = train.loc[series_mask, "qty_sold"].dropna()
test_series = test.loc[
    (test["retailer"] == "DG") & (test["store"] == train["store"].iloc[0]),
    "qty_sold"
].dropna()

horizon = len(test_series)

# Run all baselines
forecasts = run_all_baselines(train_series, horizon)

# Evaluate
results = []
for name, preds in forecasts.items():
    # Align lengths
    n = min(len(preds), len(test_series))
    metrics = all_metrics(test_series.values[:n], preds[:n])
    metrics["model"] = name
    results.append(metrics)

baseline_results = pd.DataFrame(results)
print(baseline_results.to_string(index=False))
```

### Baseline Models Explained

| Model | Formula | Best When |
|---|---|---|
| **Naive** | ŷₜ = yₜ₋₁ | Data is a random walk |
| **Seasonal Naive** | ŷₜ = yₜ₋₅₂ | Strong yearly seasonality |
| **Moving Average** | ŷₜ = mean(yₜ₋₁,...,yₜ₋₄) | Stable, low-variance series |
| **SES** | ŷₜ = α·yₜ₋₁ + (1-α)·ŷₜ₋₁ | Smooth trends, no seasonality |

---

## Step 7: Advanced Model Selection & Reasoning

### What

Train advanced models selected based on the data validation results from Step 3.

### Why Each Model Family

#### XGBoost / LightGBM (Gradient Boosted Trees)

**When to use**: Multi-series forecasting, tabular features, non-linear relationships.

**Why it works for retail**:
- Handles **missing values** natively (common in retail)
- Captures **feature interactions** (e.g., store × holiday × category effects)
- **Cross-learns** across series when pooled (DG store patterns inform other stores)
- Robust to **skewed distributions** and **outliers** (promotional spikes)

**When it fails**: Pure extrapolation (unprecedented trends), very short series (<30 points).

```python
from src.models.ml_models import train_xgboost, train_lightgbm, feature_importance_df
from src.config.config import DATE_COLUMN, PRIMARY_TARGET

# Define feature columns (exclude target and date)
exclude = {DATE_COLUMN, PRIMARY_TARGET, "net_sales", "retailer", "store", "product"}
feature_cols = [c for c in df_features.columns if c not in exclude and df_features[c].dtype in ["float64", "int64", "int32"]]

# Categorical columns for encoding
cat_cols = [c for c in ["retailer", "store"] if c in df_features.columns]

# Train XGBoost
xgb_model, xgb_test, xgb_encoders = train_xgboost(
    df_features, feature_cols, PRIMARY_TARGET, DATE_COLUMN, cat_cols,
    cutoff_frac=0.8,
)

# Train LightGBM
lgb_model, lgb_test, lgb_encoders = train_lightgbm(
    df_features, feature_cols, PRIMARY_TARGET, DATE_COLUMN, cat_cols,
    cutoff_frac=0.8,
)

# Feature importance
fi = feature_importance_df(xgb_model, feature_cols)
print(fi.head(15))
```

#### ARIMA / SARIMAX

**When to use**: Single interpretable series, need for statistical confidence intervals.

**Why**: Formally models autocorrelation (AR), differencing (I), and moving average (MA). SARIMAX adds seasonal terms.

**When it fails**: Multi-series (must fit individually), non-linear dynamics, exogenous interactions.

```python
# ARIMA for a single series (requires statsmodels)
try:
    from statsmodels.tsa.arima.model import ARIMA
    import warnings
    warnings.filterwarnings('ignore')

    # Fit ARIMA on aggregated DG series
    dg_series = train[train["retailer"] == "DG"].groupby("date")["qty_sold"].sum().sort_index()
    
    # If non-stationary (from Step 3), use d=1
    d = 0 if report["stationarity"]["is_stationary"] else 1
    model = ARIMA(dg_series, order=(2, d, 2))
    fitted = model.fit()
    
    forecast = fitted.forecast(steps=13)
    print(f"ARIMA({2},{d},{2}) forecast (13 periods):")
    print(forecast)
except ImportError:
    print("Install statsmodels: pip install statsmodels")
```

#### Prophet

**When to use**: Strong seasonality, holiday effects, changepoints in trend.

**Why**: Additive decomposition model designed specifically for business time series. Easy to add holiday effects and interpret components.

**When it fails**: Zero-inflated series, very large numbers of series (slow), complex feature interactions.

```python
# Prophet for a single series (requires prophet)
try:
    from prophet import Prophet
    
    # Prophet needs columns named 'ds' and 'y'
    dg_prophet = train[train["retailer"] == "DG"].groupby("date")["qty_sold"].sum().reset_index()
    dg_prophet.columns = ["ds", "y"]
    
    # Apply log-transform if skewed (from Step 3)
    import numpy as np
    if report["distribution"]["skewness"] > 1.0:
        dg_prophet["y"] = np.log1p(dg_prophet["y"])
        print("Applied log1p transform due to skewness")
    
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False)
    m.fit(dg_prophet)
    
    future = m.make_future_dataframe(periods=13, freq="W-MON")
    forecast = m.predict(future)
    
    # If log-transformed, reverse it
    if report["distribution"]["skewness"] > 1.0:
        forecast["yhat"] = np.expm1(forecast["yhat"])
    
    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(13))
except ImportError:
    print("Install prophet: pip install prophet")
```

#### LSTM / Neural Networks

**When to use**: Large datasets (500+ points per series, 50+ series), complex temporal dependencies.

**Why**: Can learn arbitrary temporal patterns without explicit feature engineering. Captures long-range dependencies.

**When it fails**: Small datasets (overfits), expensive to tune, hard to interpret.

```python
# LSTM example (requires tensorflow/pytorch)
# NOTE: Only use when data volume justifies it (see Step 3 recommendations)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    
    # Prepare sequence data for a single series
    dg_series = train[train["retailer"] == "DG"].groupby("date")["qty_sold"].sum().sort_index().values
    
    # Normalize (REQUIRED for neural networks)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(dg_series.reshape(-1, 1))
    
    # Create sequences (lookback window)
    lookback = 52  # 1 year of weekly data
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Build model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=50, batch_size=32, verbose=0)
    
    print(f"LSTM trained on {len(X)} sequences (lookback={lookback})")
except ImportError:
    print("Install tensorflow: pip install tensorflow")
```

---

## Step 8: Hyperparameter Tuning

### What

Optimize model parameters to find the best configuration for this specific dataset.

### Why

Default parameters are generic. Tuning typically improves MAE by **10-30%** for tree-based models. The key is tuning the right parameters (not all of them).

### Code

```python
# Optuna-based tuning for XGBoost (recommended)
try:
    import optuna
    from xgboost import XGBRegressor
    from src.evaluation.metrics import mae
    
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "random_state": 42,
            "n_jobs": -1,
        }
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_test)
        return mae(y_test.values, preds)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    print(f"Best MAE: {study.best_value:.2f}")
    print(f"Best params: {study.best_params}")
except ImportError:
    print("Install optuna: pip install optuna")
```

### Key Parameters by Model

| Model | Critical Params | Typical Range | Effect |
|---|---|---|---|
| **XGBoost** | `max_depth` | 3–8 | Controls overfitting; deeper = more complex |
| | `learning_rate` | 0.01–0.3 | Lower = more trees needed but better generalization |
| | `n_estimators` | 100–500 | More trees = better fit but slower |
| | `min_child_weight` | 1–20 | Higher = more conservative splits |
| **ARIMA** | `(p, d, q)` | Auto (via AIC) | `p`=AR order, `d`=differencing, `q`=MA order |
| **Prophet** | `changepoint_prior_scale` | 0.001–0.5 | Flexibility of trend changes |
| | `seasonality_prior_scale` | 0.01–10 | Strength of seasonal components |
| **LSTM** | `lookback` | 26–52 | How far back the network sees |
| | `learning_rate` | 1e-4–1e-2 | Too high = unstable, too low = slow |

---

## Step 9: Backtesting & Evaluation

### What

Simulate real forecasting conditions using rolling-origin cross-validation.

### Why

A single train/test split gives you **one** data point about model quality. Rolling-origin backtesting gives you **multiple** realistic evaluations across different time periods, showing how the model performs under varying conditions (trend, seasonal phases, etc.).

### Code

```python
from src.evaluation.backtesting import (
    BacktestConfig, rolling_origin_backtest, summarize_backtest
)
import numpy as np

# Configure backtest
config = BacktestConfig(
    date_col="date",
    target_col="qty_sold",
    horizon=13,       # 13-week forecast horizon
    n_splits=4,       # 4 rolling folds
    min_train_periods=52,  # At least 1 year of training
)

# Define a forecast function for XGBoost
def xgb_forecast(train_df, test_df):
    # (Re-train or use pre-trained model)
    X = test_df[feature_cols].fillna(0)
    return np.maximum(0, xgb_model.predict(X))

# Run backtest
results = rolling_origin_backtest(df_features, xgb_forecast, config)

# Summarize
summary = summarize_backtest(results)
print(summary.to_string(index=False))
```

### Metrics Explained

| Metric | Formula Intuition | Use When |
|---|---|---|
| **MAE** | Average absolute error | General-purpose, interpretable ("off by X units") |
| **RMSE** | Penalizes large errors more | When big errors are costly (stockouts) |
| **MAPE** | Percentage error | Comparing across different scales |
| **sMAPE** | Symmetric percentage | Avoids MAPE's asymmetry issue |
| **WAPE** | Weighted error by volume | Business KPI (larger products matter more) |
| **Bias** | Average over/under prediction | Detecting systematic forecast direction |

### Why Multiple Metrics?

No single metric tells the whole story:
- A model with low MAE but high Bias systematically over-predicts
- A model with low MAPE but high RMSE is good on average but has catastrophic misses
- WAPE is the most business-relevant (weighted by volume)

---

## Step 10: Model Comparison & Final Selection

### What

Compare all models side-by-side and select the best one (or ensemble) for deployment.

### Why

Different models may excel at different segments. The final selection should consider:
- **Overall accuracy** across all series
- **Segment-level performance** (does it fail for one retailer?)
- **Computational cost** (can it retrain weekly?)
- **Interpretability** (can stakeholders understand it?)

### Code

```python
from src.evaluation.metrics import all_metrics, evaluate_by_segment
from src.visualization.plots import plot_model_comparison, plot_forecast_vs_actual
import pandas as pd

# Collect results from all models
model_results = []

# XGBoost results
xgb_metrics = all_metrics(
    xgb_test["qty_sold"].fillna(0).values,
    xgb_test["prediction"].values,
)
xgb_metrics["model"] = "XGBoost"
model_results.append(xgb_metrics)

# LightGBM results
lgb_metrics = all_metrics(
    lgb_test["qty_sold"].fillna(0).values,
    lgb_test["prediction"].values,
)
lgb_metrics["model"] = "LightGBM"
model_results.append(lgb_metrics)

# Add baseline results
for _, row in baseline_results.iterrows():
    model_results.append(row.to_dict())

# Compare
comparison = pd.DataFrame(model_results)
print(comparison.sort_values("MAE").to_string(index=False))

# Visualize
fig = plot_model_comparison(comparison, metric="MAE")
fig.show()

# Segment-level evaluation
segment_metrics = evaluate_by_segment(
    xgb_test, "qty_sold", "prediction", ["retailer"]
)
print("\nXGBoost by retailer:")
print(segment_metrics.to_string(index=False))

# Forecast vs Actual plot
fig = plot_forecast_vs_actual(xgb_test, actual_col="qty_sold", group_col="retailer")
fig.show()
```

### Decision Framework

| Criterion | Weight | How to Evaluate |
|---|---|---|
| **Accuracy (WAPE)** | 40% | Lower is better; compare across all segments |
| **Robustness (backtest variance)** | 25% | Low variance across folds = stable model |
| **Segment fairness** | 20% | No retailer should have >2x worse error |
| **Interpretability** | 10% | Can you explain predictions to stakeholders? |
| **Speed** | 5% | Can it retrain and forecast in <1 hour? |

### Feature Importance Analysis

```python
from src.visualization.plots import plot_feature_importance
from src.models.ml_models import feature_importance_df

# Get top features
fi = feature_importance_df(xgb_model, feature_cols)
fig = plot_feature_importance(fi, top_n=20)
fig.show()

# Identify if lag features dominate (expected for time series)
lag_importance = fi[fi["feature"].str.startswith("lag_")]["importance"].sum()
total = fi["importance"].sum()
print(f"\nLag features account for {lag_importance/total:.1%} of importance")
```

---

## Summary: Reproducibility Checklist

| Step | Key Output | Status |
|---|---|---|
| 1. Ingestion | Unified DataFrame | ✅ `harmonize_to_unified()` |
| 2. EDA | Visual understanding of data | ✅ `plots.py` |
| 3. Validation | Distribution report + model recommendation | ✅ `validation.py` |
| 4. Features | Engineered DataFrame with lags, rolling stats | ✅ `build_features()` |
| 5. Splitting | Time-based train/test | ✅ Never random |
| 6. Baselines | Performance floor | ✅ `run_all_baselines()` |
| 7. Models | ARIMA, Prophet, XGBoost, LightGBM, LSTM | ✅ Best fit per data |
| 8. Tuning | Optimized hyperparameters | ✅ Optuna |
| 9. Backtest | Rolling-origin evaluation | ✅ `rolling_origin_backtest()` |
| 10. Selection | Final model + reasoning | ✅ Evidence-based |

---

## Dependencies

```bash
pip install pandas numpy scipy scikit-learn
pip install statsmodels   # for ARIMA + stationarity tests
pip install xgboost       # for XGBoost
pip install lightgbm      # for LightGBM
pip install prophet       # for Prophet (optional)
pip install tensorflow    # for LSTM (optional)
pip install optuna        # for hyperparameter tuning (optional)
pip install plotly        # for visualization
```
