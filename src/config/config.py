"""
Project-wide configuration.

Updated with real database schemas: DG, S&F, WG.
All schema-specific settings reference the actual column names
from the three databases.
"""

from __future__ import annotations
import pathlib

# ---------------------------------------------------------------------------
# Forecasting grain
# ---------------------------------------------------------------------------
DEFAULT_GRAIN: str = "weekly"          # aggregate daily data to weekly
FREQUENCY_ALIAS: str = "W-MON"         # pandas frequency

# ---------------------------------------------------------------------------
# Database identifiers (the 3 retailers)
# ---------------------------------------------------------------------------
DATABASES = ["DG", "SF", "WG"]

# ---------------------------------------------------------------------------
# Unified internal column names (used across the pipeline)
# ---------------------------------------------------------------------------
DATE_COLUMN: str = "date"
RETAILER_COLUMN: str = "retailer"      # DG / SF / WG
STORE_COLUMN: str = "store"
PRODUCT_COLUMN: str = "product"
CATEGORY_COLUMN: str = "category"
DEPARTMENT_COLUMN: str = "department"
CITY_COLUMN: str = "city"
STATE_COLUMN: str = "state"

# Primary forecasting target (unified name)
PRIMARY_TARGET: str = "qty_sold"
SECONDARY_TARGETS: list[str] = ["net_sales"]

# ---------------------------------------------------------------------------
# Retailer (database) profiles for synthetic data generation
# ---------------------------------------------------------------------------
RETAILER_PROFILES: dict = {
    "DG": {
        "description": "Dollar General — stable growth, strong holiday seasonality, large store footprint",
        "trend_slope": 0.8,
        "seasonality_strength": 1.5,
        "noise_level": "low",
        "base_units": 500,
        "grain": "store × product × day",
        "n_stores": 50,
        "n_products": 30,
    },
    "SF": {
        "description": "S&F — flat trend, heavy promotional spikes, transaction-level detail",
        "trend_slope": 0.05,
        "seasonality_strength": 0.5,
        "noise_level": "medium",
        "base_units": 350,
        "promo_frequency": "high",
        "grain": "transaction-level",
        "n_stores": 25,
        "n_products": 20,
    },
    "WG": {
        "description": "Walgreens — summer seasonality, category-level aggregation, regional volatility",
        "trend_slope": 0.2,
        "seasonality_strength": 2.0,
        "noise_level": "high",
        "base_units": 300,
        "peak_season": "summer",
        "grain": "store × category × day",
        "n_stores": 40,
        "n_categories": 10,
    },
}

# ---------------------------------------------------------------------------
# DG-specific product hierarchy
# ---------------------------------------------------------------------------
DG_DEPARTMENTS = {
    1: {"desc": "Consumables",     "classes": [(101, "Snacks"), (102, "Beverages"), (103, "Candy")]},
    2: {"desc": "Health & Beauty", "classes": [(201, "OTC Medicine"), (202, "Personal Care")]},
    3: {"desc": "Home Products",   "classes": [(301, "Cleaning"), (302, "Paper Products")]},
    4: {"desc": "Seasonal",        "classes": [(401, "Holiday"), (402, "Outdoor")]},
    5: {"desc": "Apparel",         "classes": [(501, "Basics"), (502, "Accessories")]},
}

# ---------------------------------------------------------------------------
# WG categories
# ---------------------------------------------------------------------------
WG_CATEGORIES = [
    "Pharmacy", "Health & Wellness", "Beauty", "Personal Care",
    "Household", "Consumables", "Photo", "Seasonal",
    "General Merchandise", "Baby & Child",
]

# ---------------------------------------------------------------------------
# DG geography (sample cities/states for mock data)
# ---------------------------------------------------------------------------
DG_LOCATIONS = [
    ("Houston", "TX"), ("Dallas", "TX"), ("Atlanta", "GA"),
    ("Nashville", "TN"), ("Charlotte", "NC"), ("Raleigh", "NC"),
    ("Birmingham", "AL"), ("Memphis", "TN"), ("Jacksonville", "FL"),
    ("Louisville", "KY"), ("Richmond", "VA"), ("Columbus", "OH"),
]

# ---------------------------------------------------------------------------
# Data date range (synthetic)
# ---------------------------------------------------------------------------
SYNTH_START_DATE: str = "2021-01-04"
SYNTH_END_DATE: str = "2024-12-30"

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
FORECAST_HORIZON_WEEKS: int = 13
BACKTEST_STEPS: int = 4
TRAIN_CUTOFF_FRAC: float = 0.8

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
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
