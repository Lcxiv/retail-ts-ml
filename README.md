# 🛒 Time Series Retail Sales ML Framework

An end-to-end **time series machine learning pipeline** for analyzing and forecasting retail sales trends across multiple retailers and selling objects.

## 📚 ML Pipeline Guide

**→ [Complete Step-by-Step Guide](docs/ml_pipeline_guide.md)** — 10-step reproducible walkthrough with code snippets and reasoning.

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
├── docs/
│   ├── ml_pipeline_guide.md    # ⭐ Step-by-step ML guide with reasoning
│   └── databases_schema_reference.md
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
│   ├── data/             # Mock generator, schema, validation
│   ├── features/         # Feature engineering pipeline
│   ├── models/           # Baseline and ML models
│   ├── evaluation/       # Metrics and backtesting
│   ├── visualization/    # Plotly charts
│   └── utils/            # Helper utilities
│
├── tests/
│   ├── test_schema.py
│   ├── test_feature_engineering.py
│   └── test_validation.py
│
├── outputs/
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

# 5. Run data validation
python -c "from src.data.validation import generate_validation_report, print_validation_report; import pandas as pd; df = pd.read_parquet('data/synthetic/synthetic_dg.parquet'); from src.data.schema import DG_SCHEMA, harmonize_to_unified; report = generate_validation_report(harmonize_to_unified(df, DG_SCHEMA)); print_validation_report(report)"

# 6. Run tests
python -m pytest tests/ -v

# 7. Launch Jupyter
jupyter lab
```

## Execution Phases

| Phase | Name | Description | Guide Step |
|-------|------|-------------|------------|
| 0 | Planning | Gather schema, confirm grain & targets | — |
| 1 | Synthetic Data | Generate realistic mock retail sales data | [Step 1](docs/ml_pipeline_guide.md#step-1-data-ingestion--harmonization) |
| 2 | EDA | Decompose series, profile retailers & objects | [Step 2](docs/ml_pipeline_guide.md#step-2-exploratory-data-analysis) |
| 3 | Validation | Test distributions, check stationarity | [Step 3](docs/ml_pipeline_guide.md#step-3-data-validation--distribution-assessment) |
| 4 | Feature Engineering | Lags, rolling stats, hierarchy, events | [Step 4](docs/ml_pipeline_guide.md#step-4-feature-engineering) |
| 5 | Baseline Modeling | Naive, seasonal naive, moving average, ETS | [Step 6](docs/ml_pipeline_guide.md#step-6-baseline-models) |
| 6 | Advanced Modeling | ARIMA, Prophet, XGBoost, LightGBM, LSTM | [Step 7](docs/ml_pipeline_guide.md#step-7-advanced-model-selection--reasoning) |
| 7 | Evaluation | Rolling backtest, segment-level metrics | [Step 9](docs/ml_pipeline_guide.md#step-9-backtesting--evaluation) |

## Tech Stack

- **Core**: Python, pandas, numpy, scipy
- **Visualization**: plotly, matplotlib
- **ML**: scikit-learn, xgboost, lightgbm
- **Time Series**: statsmodels, prophet (optional)
- **Deep Learning**: tensorflow (optional)
- **Storage**: pyarrow / parquet
- **Tuning**: optuna (optional)
- **Notebooks**: Jupyter

## Status

> **Current Phase**: Phase 4 — Pipeline validated, ML guide complete
>
> All source modules fixed and tested. See [ML Pipeline Guide](docs/ml_pipeline_guide.md) for the complete step-by-step walkthrough.

## Contributing

See [AGENTS.md](AGENTS.md) for agent roles, execution rules, and orchestration guide.
