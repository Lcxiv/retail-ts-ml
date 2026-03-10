# 🛒 Time Series Retail Sales ML Framework

An end-to-end **time series machine learning pipeline** for analyzing and forecasting retail sales trends across multiple retailers and selling objects.

## Purpose

This project models retail sales behavior over time, compares trend dynamics across retailers, identifies seasonality and anomalies, and generates accurate forecasts at multiple aggregation levels.

## Business Questions Answered

- Which retailers are growing or declining over time?
- Which selling objects show the strongest positive/negative trend?
- Are there retailer-specific seasonal patterns?
- Which products are volatile vs. stable?
- Can future sales be forecasted reliably by retailer, object, category, or region?
- Which time periods show unusual spikes or drops?

## Project Architecture

```
retail-ts-ml/
│
├── data/
│   ├── raw/              # Real data (TBD when schema is provided)
│   ├── synthetic/        # Generated mock data
│   ├── processed/        # Cleaned & transformed data
│   └── external/         # Holiday calendars, economic signals, etc.
│
├── notebooks/
│   ├── 01_schema_design.ipynb
│   ├── 02_mock_data_generation.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_modeling_baselines.ipynb
│   ├── 06_modeling_advanced.ipynb
│   └── 07_evaluation_and_insights.ipynb
│
├── src/
│   ├── config/           # Schema config, grain, targets
│   ├── data/             # Mock generator, schema definitions
│   ├── features/         # Feature engineering pipeline
│   ├── models/           # Baseline and ML models
│   ├── evaluation/       # Metrics and backtesting
│   ├── visualization/    # Plotly charts
│   └── utils/            # Helper utilities
│
├── outputs/
│   ├── plots/
│   ├── forecasts/
│   ├── diagnostics/
│   └── reports/
│
├── tests/
├── requirements.txt
├── README.md
└── AGENTS.md
```

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Lcxiv/retail-ts-ml.git
cd retail-ts-ml

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate synthetic data
python -m src.data.mock_generator

# 5. Launch Jupyter
jupyter lab
```

## Execution Phases

| Phase | Name | Description |
|-------|------|-------------|
| 0 | Planning | Gather schema, confirm grain & targets |
| 1 | Synthetic Data | Generate realistic mock retail sales data |
| 2 | EDA | Decompose series, profile retailers & objects |
| 3 | Feature Engineering | Lags, rolling stats, hierarchy, events |
| 4 | Baseline Modeling | Naive, seasonal naive, moving average, ETS |
| 5 | Advanced Modeling | ARIMA, Prophet, XGBoost, LightGBM |
| 6 | Evaluation | Rolling backtest, segment-level metrics |
| 7 | Packaging | Document assumptions, real-data transition |

## Tech Stack

- **Core**: Python, pandas, numpy
- **Visualization**: matplotlib, plotly
- **ML**: scikit-learn, xgboost, lightgbm
- **Time Series**: statsmodels, prophet
- **Storage**: pyarrow / parquet
- **Notebooks**: Jupyter
- **Optional**: mlflow, optuna, streamlit

## Status

> **Current Phase**: Pre-schema planning (Phase 0)
>
> Awaiting final column list from user to finalize schema, grain, and target variable.

## Contributing

See [AGENTS.md](AGENTS.md) for agent roles, execution rules, and orchestration guide.
