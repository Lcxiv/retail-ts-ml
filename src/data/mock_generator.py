"""
Synthetic Retail Sales Data Generator.

Generates realistic mock data for 3 databases:
  - DG:  daily store × product (UPC/SKU), departments, classes, geography
  - S&F: transaction-level with integer dates
  - WG:  daily store × category (aggregated)

All column names match the actual database schemas.

Usage:
    python -m src.data.mock_generator
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.config import (
    RETAILER_PROFILES,
    SYNTH_START_DATE,
    SYNTH_END_DATE,
    DATA_SYNTHETIC_DIR,
    DG_DEPARTMENTS,
    DG_LOCATIONS,
    WG_CATEGORIES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

NP_SEED = 42
rng = np.random.default_rng(NP_SEED)
random.seed(NP_SEED)


# ---------------------------------------------------------------------------
# Shared helper functions
# ---------------------------------------------------------------------------

def _daily_dates() -> pd.DatetimeIndex:
    """Daily date range for the synthetic period."""
    return pd.date_range(start=SYNTH_START_DATE, end=SYNTH_END_DATE, freq="D")


def _trend(n: int, slope: float) -> np.ndarray:
    return slope * np.arange(n)


def _seasonality(dates: pd.DatetimeIndex, strength: float, peak: str | None = None) -> np.ndarray:
    doy = dates.dayofyear.to_numpy(dtype=float)
    if peak == "summer":
        return strength * 60 * np.exp(-0.5 * ((doy - 196) / 40) ** 2)
    # holiday peak (day ~340) + mild summer bump (day ~175)
    return (
        strength * 80 * np.exp(-0.5 * ((doy - 340) / 30) ** 2)
        + strength * 30 * np.exp(-0.5 * ((doy - 175) / 40) ** 2)
    )


def _noise(n: int, level: str, base: float) -> np.ndarray:
    scale_map = {"low": 0.03, "medium": 0.08, "high": 0.18}
    return rng.normal(0, scale_map.get(level, 0.08) * base, n)


def _promo_spikes(n: int, freq: str, base: float) -> np.ndarray:
    spikes = np.zeros(n)
    rate = {"high": 0.12, "medium": 0.06, "low": 0.02}.get(freq, 0.04)
    idx = rng.choice(n, size=int(n * rate), replace=False)
    spikes[idx] = rng.uniform(0.5 * base, 1.5 * base, len(idx))
    return spikes


def _anomalies(n: int, base: float) -> np.ndarray:
    a = np.zeros(n)
    k = max(1, int(n * 0.01))
    idx = rng.choice(n, size=k, replace=False)
    a[idx] = rng.choice([-1, 1], k) * rng.uniform(0.4 * base, 0.9 * base, k)
    return a


def _inject_missing(arr: np.ndarray, frac: float = 0.02) -> np.ndarray:
    arr = arr.copy()
    idx = rng.choice(len(arr), size=int(len(arr) * frac), replace=False)
    arr[idx] = np.nan
    return arr


# ---------------------------------------------------------------------------
# DG: store × product × day
# ---------------------------------------------------------------------------

def generate_dg() -> pd.DataFrame:
    """Generate synthetic DG data with actual column names."""
    profile = RETAILER_PROFILES["DG"]
    dates = _daily_dates()
    n_dates = len(dates)
    n_stores = profile["n_stores"]
    n_products = profile["n_products"]

    log.info("Generating DG: %d days × %d stores × %d products", n_dates, n_stores, n_products)

    # Build product catalog
    products = []
    for dept_nbr, dept_info in DG_DEPARTMENTS.items():
        for class_nbr, class_desc in dept_info["classes"]:
            for i in range(max(1, n_products // len(DG_DEPARTMENTS) // len(dept_info["classes"]))):
                pid = len(products)
                products.append({
                    "UPC": f"0{4900000000 + pid:010d}",
                    "SKU": f"SKU-{pid:05d}",
                    "P_SKU": f"PSKU-{pid:05d}",
                    "DESCRIPTION": f"{class_desc} Item {i+1}",
                    "CLASS_NBR": class_nbr,
                    "CLASS_DESC": class_desc,
                    "DEPARTMENT_NBR": dept_nbr,
                    "DEPARTMENT_DESC": dept_info["desc"],
                })

    # Assign stores to locations
    stores = []
    for s in range(1, n_stores + 1):
        city, state = DG_LOCATIONS[s % len(DG_LOCATIONS)]
        stores.append({"STORE": s, "CITY": city, "STATE": state})

    # Sample a subset (not all stores carry all products every day)
    sample_size = min(200_000, n_dates * n_stores * len(products))  # cap for speed
    rows = []
    base = profile["base_units"]

    for product in products:
        for store in random.sample(stores, min(10, len(stores))):
            trend = _trend(n_dates, profile["trend_slope"])
            season = _seasonality(dates, profile["seasonality_strength"])
            noise = _noise(n_dates, profile["noise_level"], base)
            promo = _promo_spikes(n_dates, "low", base)
            qty = np.maximum(0, base + trend + season + noise + promo + _anomalies(n_dates, base))
            qty = _inject_missing(qty)
            net_sales = qty * rng.uniform(1.0, 15.0, n_dates)
            trans_ct = np.maximum(1, (qty * rng.uniform(0.3, 0.8, n_dates)).astype(int))

            for day_idx in range(n_dates):
                if rng.random() > 0.3:  # ~70% fill rate to simulate realistic sparsity
                    rows.append({
                        "DATE": dates[day_idx],
                        "STORE": store["STORE"],
                        "CITY": store["CITY"],
                        "STATE": store["STATE"],
                        "OPEN_CLOSED_CODE": "O" if rng.random() > 0.02 else "C",
                        **{k: product[k] for k in ["UPC", "SKU", "P_SKU", "DESCRIPTION",
                                                    "CLASS_NBR", "CLASS_DESC",
                                                    "DEPARTMENT_NBR", "DEPARTMENT_DESC"]},
                        "TRANS_CT": int(trans_ct[day_idx]),
                        "QTY_SOLD": round(float(qty[day_idx]), 0) if not np.isnan(qty[day_idx]) else None,
                        "NET_SALES": round(float(net_sales[day_idx]), 2) if not np.isnan(qty[day_idx]) else None,
                    })
                    if len(rows) >= sample_size:
                        break
                if len(rows) >= sample_size:
                    break
            if len(rows) >= sample_size:
                break
        if len(rows) >= sample_size:
            break

    df = pd.DataFrame(rows)
    log.info("DG dataset: %s rows, %d columns", f"{len(df):,}", len(df.columns))
    return df


# ---------------------------------------------------------------------------
# S&F: transaction-level
# ---------------------------------------------------------------------------

def generate_sf() -> pd.DataFrame:
    """Generate synthetic S&F data with actual column names."""
    profile = RETAILER_PROFILES["SF"]
    dates = _daily_dates()
    n_dates = len(dates)
    n_stores = profile["n_stores"]
    n_products = profile["n_products"]

    log.info("Generating S&F: %d days × %d stores × %d products (transaction-level)", n_dates, n_stores, n_products)

    # Product catalog
    upc_numbers = [4900000000 + i for i in range(n_products)]
    item_numbers = [f"ITEM-{i:05d}" for i in range(n_products)]

    base = profile["base_units"]
    sample_cap = 200_000
    rows = []
    txn_counter = 0

    for store in range(1, n_stores + 1):
        for day_idx in range(n_dates):
            if rng.random() > 0.25:  # some days may have no data
                n_txns = max(1, int(rng.poisson(8)))  # avg 8 transactions per store-day
                for _ in range(n_txns):
                    txn_counter += 1
                    prod_idx = rng.integers(0, n_products)
                    item_qty = max(1, int(rng.poisson(3)))
                    unit_price = round(float(rng.uniform(2.0, 50.0)), 5)
                    total_price = round(unit_price * item_qty, 5)

                    rows.append({
                        "TRANSACTIONDATE": int(dates[day_idx].strftime("%Y%m%d")),
                        "TRANSACTIONKEY": f"TXN-{txn_counter:08d}",
                        "POSCARDNUMBER": f"CARD-{rng.integers(100000, 999999)}",
                        "STORENUMBER": store,
                        "UPCNUMBER": upc_numbers[prod_idx],
                        "ITEMNUMBER": item_numbers[prod_idx],
                        "ITEMTOTALSALESPRICEAMOUNT": unit_price,
                        "ITEMQUANTITY": item_qty,
                        "TOTALNETPRICEAMOUNT": total_price,
                        "FILENAME": f"SF_FEED_{dates[day_idx].strftime('%Y%m%d')}.csv",
                        "FILEDATE": int(dates[day_idx].strftime("%Y%m%d")),
                        "ROWNUMBER": txn_counter,
                    })
                    if len(rows) >= sample_cap:
                        break
            if len(rows) >= sample_cap:
                break
        if len(rows) >= sample_cap:
            break

    df = pd.DataFrame(rows)
    log.info("S&F dataset: %s rows, %d columns", f"{len(df):,}", len(df.columns))
    return df


# ---------------------------------------------------------------------------
# WG: store × category × day
# ---------------------------------------------------------------------------

def generate_wg() -> pd.DataFrame:
    """Generate synthetic WG data with actual column names."""
    profile = RETAILER_PROFILES["WG"]
    dates = _daily_dates()
    n_dates = len(dates)
    n_stores = profile["n_stores"]
    categories = WG_CATEGORIES

    log.info("Generating WG: %d days × %d stores × %d categories", n_dates, n_stores, len(categories))

    base = profile["base_units"]
    rows = []

    for store in range(1, n_stores + 1):
        for cat in categories:
            cat_base = base * rng.uniform(0.5, 2.0)
            trend = _trend(n_dates, profile["trend_slope"])
            season = _seasonality(dates, profile["seasonality_strength"], profile.get("peak_season"))
            noise = _noise(n_dates, profile["noise_level"], cat_base)
            qty = np.maximum(0, cat_base + trend + season + noise + _anomalies(n_dates, cat_base))
            qty = _inject_missing(qty)
            avg_price = rng.uniform(5.0, 30.0, n_dates)
            sales = qty * avg_price

            for day_idx in range(n_dates):
                rows.append({
                    "DATE": dates[day_idx],
                    "STORE": store,
                    "CATEGORY": cat,
                    "CAT_QTY_SOLD": round(float(qty[day_idx]), 0) if not np.isnan(qty[day_idx]) else None,
                    "CAT_NET_SALES": round(float(sales[day_idx]), 5) if not np.isnan(qty[day_idx]) else None,
                })

    df = pd.DataFrame(rows)
    log.info("WG dataset: %s rows, %d columns", f"{len(df):,}", len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Main generator — generates all 3 databases
# ---------------------------------------------------------------------------

def generate_all(save: bool = True) -> dict[str, pd.DataFrame]:
    """
    Generate synthetic datasets for all 3 databases.

    Returns
    -------
    dict of {db_name: DataFrame}
    """
    generators = {
        "DG": generate_dg,
        "SF": generate_sf,
        "WG": generate_wg,
    }

    results = {}
    for name, gen_fn in generators.items():
        df = gen_fn()
        results[name] = df

        if save:
            csv_path = DATA_SYNTHETIC_DIR / f"synthetic_{name.lower()}.csv"
            parquet_path = DATA_SYNTHETIC_DIR / f"synthetic_{name.lower()}.parquet"
            df.to_csv(csv_path, index=False)
            df.to_parquet(parquet_path, index=False)
            log.info("Saved %s to %s and %s", name, csv_path, parquet_path)

    log.info("\n=== Summary ===")
    for name, df in results.items():
        log.info("%s: %s rows × %d cols | columns: %s", name, f"{len(df):,}", len(df.columns), list(df.columns))

    return results


if __name__ == "__main__":
    generate_all(save=True)
