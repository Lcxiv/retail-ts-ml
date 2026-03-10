# PROJECT_MAP.md — retail-ts-ml

Central registry of all core components. Read this first in every session.

> Last updated: 2026-03-10

## Architecture Overview

```
retail-ts-ml/
├── src/config/config.py        → Schema config, grain, targets, retailer profiles
├── src/data/schema.py          → DatabaseSchema definitions (DG/SF/WG), harmonize_to_unified()
├── src/data/mock_generator.py  → Synthetic data generator for all 3 databases
├── src/data/validation.py      → Distribution analysis, stationarity, model recommendations
├── src/features/feature_engineering.py → Lags, rolling stats, calendar, hierarchical features
├── src/models/baseline_models.py      → Naive, Seasonal Naive, MA, SES
├── src/models/ml_models.py            → XGBoost, LightGBM training + feature importance
├── src/evaluation/metrics.py          → MAE, RMSE, MAPE, sMAPE, WAPE, Bias
├── src/evaluation/backtesting.py      → Rolling-origin cross-validation engine
├── src/visualization/plots.py         → Plotly charts (trend, heatmap, comparison, etc.)
├── src/utils/helpers.py               → Date utils, DataFrame profiling, timing decorator
├── docs/ml_pipeline_guide.md          → ⭐ 10-step ML guide with reasoning
├── notebooks/00_full_pipeline.ipynb   → Runnable end-to-end notebook
├── tests/test_schema.py               → Schema harmonization tests
├── tests/test_feature_engineering.py  → Feature creation + leakage tests
├── tests/test_validation.py           → Distribution/stationarity/model recommendation tests
├── PROJECT_MAP.md                     → This file (central registry)
└── task_active.md                     → Current tasks and priorities
```

## Data Flow

```
Raw DB data (DG/SF/WG)
  → harmonize_to_unified()        [schema.py]
  → build_features()              [feature_engineering.py]
  → _time_split()                 [ml_models.py]
  → train_xgboost/lightgbm()      [ml_models.py]
  → rolling_origin_backtest()     [backtesting.py]
  → all_metrics()                 [metrics.py]
  → plot_forecast_vs_actual()     [plots.py]
```

## Key Constants

| Constant | Value | Defined In |
|---|---|---|
| `PRIMARY_TARGET` | `"qty_sold"` | `config.py` |
| `DATE_COLUMN` | `"date"` | `config.py` |
| `DATABASES` | `["DG", "SF", "WG"]` | `config.py` |
| `DEFAULT_GRAIN` | `"weekly"` | `config.py` |
| `FORECAST_HORIZON_WEEKS` | `13` | `config.py` |
| `TRAIN_CUTOFF_FRAC` | `0.8` | `config.py` |

## Database Schemas

| DB | Grain | Date Key | Store Key | Qty Target | Sales Target |
|---|---|---|---|---|---|
| DG | store × product × day | `DATE` | `STORE` | `QTY_SOLD` | `NET_SALES` |
| SF | transaction-level | `TRANSACTIONDATE` (int) | `STORENUMBER` | `ITEMQUANTITY` | `TOTALNETPRICEAMOUNT` |
| WG | store × category × day | `DATE` | `STORE` | `CAT_QTY_SOLD` | `CAT_NET_SALES` |
