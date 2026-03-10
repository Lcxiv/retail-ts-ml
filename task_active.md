# Active Tasks — retail-ts-ml

> Updated: 2026-03-10

## Current Phase: Pipeline Validated (Phase 4)

## Immediate Next Steps

- [ ] Merge PR #2 (`feat/ml-pipeline-guide` → `main`)
- [ ] Clone repo locally and run `python -m pytest tests/ -v` to verify
- [ ] Run `notebooks/00_full_pipeline.ipynb` end-to-end
- [ ] Generate synthetic data and run validation report

## Upcoming Work

- [ ] Fill in notebook stubs (01-07) with content from the ML guide
- [ ] Add real data connection when Snowflake credentials are available
- [ ] Add Prophet and LSTM models to `ml_models.py` (currently separate snippets in guide)
- [ ] Add MLflow experiment tracking integration
- [ ] Add Streamlit dashboard for interactive forecasting

## Known Issues

- `plots.py` reference columns that depend on harmonized data — if raw data is passed, some plots will error
- `feature_engineering.py` defaults assume all 3 group cols exist; WG data has no `product` column
- Notebook stubs (01-07) still have placeholder content

## Completed (this session)

- [x] Fixed broken `OBJECT_COLUMN` import in `feature_engineering.py`
- [x] Fixed `backtesting.py` target_col default
- [x] Fixed `plots.py` stale column references
- [x] Created `src/data/validation.py` (stationarity, distribution, outliers, model recs)
- [x] Created `docs/ml_pipeline_guide.md` (10-step guide)
- [x] Created `notebooks/00_full_pipeline.ipynb` (runnable notebook)
- [x] Added test suites (schema, features, validation)
- [x] Updated README + requirements.txt
