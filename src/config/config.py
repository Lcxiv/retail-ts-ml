"""
Project-wide configuration.

All schema assumptions live here so they can be swapped out
when the real dataset columns are provided.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Forecasting grain
# ---------------------------------------------------------------------------
# Options: "weekly" | "monthly" | "daily"
DEFAULT_GRAIN: str = "weekly"

# ---------------------------------------------------------------------------
# Date / time
# ---------------------------------------------------------------------------
DATE_COLUMN: str = "date"              # placeholder — update when real schema arrives
FREQUENCY_ALIAS: str = "W-MON"         # pandas frequency string for weekly starting Monday

# ---------------------------------------------------------------------------
# Entity columns (placeholders)
# ---------------------------------------------------------------------------
RETAILER_COLUMN: str = "retailer_id"
OBJECT_COLUMN: str = "object_id"
CATEGORY_COLUMN: str = "category"
BRAND_COLUMN: str = "brand"
REGION_COLUMN: str = "region"

# ---------------------------------------------------------------------------
# Target variable
# ---------------------------------------------------------------------------
# Primary target for forecasting — change once real schema is confirmed
PRIMARY_TARGET: str = "units_sold"

# Secondary targets (treated as features or supporting outputs)
SECONDARY_TARGETS: list[str] = ["revenue", "margin"]

# ---------------------------------------------------------------------------
# Entity definitions (used in mock generation)
# ---------------------------------------------------------------------------
RETAILER_PROFILES: dict = {
    "retailer_A": {
        "description": "Stable growth with strong holiday seasonality",
        "trend_slope": 0.8,
        "seasonality_strength": 1.5,
        "noise_level": "low",
        "base_units": 500,
    },
    "retailer_B": {
        "description": "Flat trend with heavy promotional spikes",
        "trend_slope": 0.05,
        "seasonality_strength": 0.5,
        "noise_level": "medium",
        "base_units": 350,
        "promo_frequency": "high",
    },
    "retailer_C": {
        "description": "Declining legacy products but stronger new-item growth",
        "trend_slope": -0.3,
        "seasonality_strength": 0.8,
        "noise_level": "medium",
        "base_units": 400,
    },
    "retailer_D": {
        "description": "Strong summer seasonality and regional volatility",
        "trend_slope": 0.2,
        "seasonality_strength": 2.0,
        "noise_level": "high",
        "base_units": 300,
        "peak_season": "summer",
    },
}

OBJECT_PROFILES: dict = {
    "obj_001": {"category": "electronics", "brand": "brand_X", "region": "north", "base_units": 200},
    "obj_002": {"category": "electronics", "brand": "brand_Y", "region": "south", "base_units": 150},
    "obj_003": {"category": "apparel",     "brand": "brand_X", "region": "east",  "base_units": 300},
    "obj_004": {"category": "apparel",     "brand": "brand_Z", "region": "west",  "base_units": 250},
    "obj_005": {"category": "home",        "brand": "brand_Y", "region": "north", "base_units": 180},
    "obj_006": {"category": "home",        "brand": "brand_Z", "region": "south", "base_units": 120},
}

# ---------------------------------------------------------------------------
# Data date range
# ---------------------------------------------------------------------------
SYNTH_START_DATE: str = "2021-01-04"   # start of first full week
SYNTH_END_DATE: str = "2024-12-30"     # ~4 years of weekly data

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
FORECAST_HORIZON_WEEKS: int = 13       # 3-month forecast horizon
BACKTEST_STEPS: int = 4               # number of rolling backtest folds
TRAIN_CUTOFF_FRAC: float = 0.8        # 80% train, 20% validation

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
import pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
DATA_SYNTHETIC_DIR = ROOT_DIR / "data" / "synthetic"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_PLOTS_DIR = ROOT_DIR / "outputs" / "plots"
OUTPUTS_FORECASTS_DIR = ROOT_DIR / "outputs" / "forecasts"
OUTPUTS_REPORTS_DIR = ROOT_DIR / "outputs" / "reports"

for _dir in [
    DATA_SYNTHETIC_DIR, DATA_PROCESSED_DIR,
    OUTPUTS_PLOTS_DIR, OUTPUTS_FORECASTS_DIR, OUTPUTS_REPORTS_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)
