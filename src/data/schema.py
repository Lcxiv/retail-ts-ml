"""
Column role mapping for 3 retail databases: DG, S&F, WG.

Each database has its own schema. The UnifiedSchema maps all three
to a common internal representation for downstream pipeline use.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


# ---------------------------------------------------------------------------
# Per-database schema definitions (actual column names)
# ---------------------------------------------------------------------------

@dataclass
class DatabaseSchema:
    """Maps a database's actual column names to semantic roles."""
    name: str
    time_key: str
    store_key: str
    qty_target: str
    sales_target: str

    # Product identifiers (may be None for aggregated DBs)
    product_keys: list[str] = field(default_factory=list)

    # Hierarchy / descriptive columns
    hierarchy: dict[str, str] = field(default_factory=dict)  # role -> column name

    # Geography
    geo_fields: list[str] = field(default_factory=list)

    # Metadata / other columns
    other_fields: list[str] = field(default_factory=list)

    # Date format notes
    date_is_integer: bool = False  # True if date stored as NUMBER

    def all_columns(self) -> list[str]:
        """All column names in this database."""
        cols = [self.time_key, self.store_key, self.qty_target, self.sales_target]
        cols += self.product_keys
        cols += list(self.hierarchy.values())
        cols += self.geo_fields
        cols += self.other_fields
        return list(dict.fromkeys(cols))  # deduplicate preserving order


# ---------------------------------------------------------------------------
# Database 1: DG
# Grain: daily × store × product (UPC/SKU level)
# ---------------------------------------------------------------------------
DG_SCHEMA = DatabaseSchema(
    name="DG",
    time_key="DATE",
    store_key="STORE",
    qty_target="QTY_SOLD",
    sales_target="NET_SALES",
    product_keys=["UPC", "SKU", "P_SKU"],
    hierarchy={
        "class_nbr": "CLASS_NBR",
        "class_desc": "CLASS_DESC",
        "dept_nbr": "DEPARTMENT_NBR",
        "dept_desc": "DEPARTMENT_DESC",
        "description": "DESCRIPTION",
    },
    geo_fields=["CITY", "STATE"],
    other_fields=["OPEN_CLOSED_CODE", "TRANS_CT"],
)


# ---------------------------------------------------------------------------
# Database 2: S&F
# Grain: transaction-level
# NOTE: TRANSACTIONDATE is NUMBER(38,0) — likely YYYYMMDD integer
# ---------------------------------------------------------------------------
SF_SCHEMA = DatabaseSchema(
    name="SF",
    time_key="TRANSACTIONDATE",
    store_key="STORENUMBER",
    qty_target="ITEMQUANTITY",
    sales_target="TOTALNETPRICEAMOUNT",
    product_keys=["UPCNUMBER", "ITEMNUMBER"],
    date_is_integer=True,
    other_fields=[
        "TRANSACTIONKEY", "POSCARDNUMBER",
        "ITEMTOTALSALESPRICEAMOUNT",
        "FILENAME", "FILEDATE", "ROWNUMBER",
    ],
)


# ---------------------------------------------------------------------------
# Database 3: WG
# Grain: daily × store × category (aggregated — no product-level detail)
# ---------------------------------------------------------------------------
WG_SCHEMA = DatabaseSchema(
    name="WG",
    time_key="DATE",
    store_key="STORE",
    qty_target="CAT_QTY_SOLD",
    sales_target="CAT_NET_SALES",
    hierarchy={"category": "CATEGORY"},
)


# All schemas for easy iteration
ALL_SCHEMAS = {"DG": DG_SCHEMA, "SF": SF_SCHEMA, "WG": WG_SCHEMA}


# ---------------------------------------------------------------------------
# Unified internal representation
# ---------------------------------------------------------------------------

@dataclass
class UnifiedColumnRoles:
    """
    Standard internal column names used by the pipeline.
    Each database's raw columns are mapped to these roles during ingestion.
    """
    date: str = "date"
    retailer: str = "retailer"         # which database / retailer the row came from
    store: str = "store"
    product: str = "product"           # primary product key (UPC where available)
    qty_sold: str = "qty_sold"         # primary target
    net_sales: str = "net_sales"       # revenue target
    category: str = "category"
    department: str = "department"
    city: str = "city"
    state: str = "state"


UNIFIED = UnifiedColumnRoles()


# ---------------------------------------------------------------------------
# Conversion utility
# ---------------------------------------------------------------------------

def convert_int_date(series: pd.Series) -> pd.Series:
    """Convert YYYYMMDD integer dates to pandas datetime."""
    return pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")


def harmonize_to_unified(
    df: pd.DataFrame,
    schema: DatabaseSchema,
) -> pd.DataFrame:
    """
    Rename a database's raw columns to the unified internal names.
    Returns a DataFrame with standardized column names.
    """
    rename_map = {
        schema.time_key: UNIFIED.date,
        schema.store_key: UNIFIED.store,
        schema.qty_target: UNIFIED.qty_sold,
        schema.sales_target: UNIFIED.net_sales,
    }

    # Map first product key to 'product'
    if schema.product_keys:
        rename_map[schema.product_keys[0]] = UNIFIED.product

    # Map hierarchy
    if "category" in schema.hierarchy:
        rename_map[schema.hierarchy["category"]] = UNIFIED.category
    if "dept_desc" in schema.hierarchy:
        rename_map[schema.hierarchy["dept_desc"]] = UNIFIED.department

    # Map geo
    if "CITY" in schema.geo_fields:
        rename_map["CITY"] = UNIFIED.city
    if "STATE" in schema.geo_fields:
        rename_map["STATE"] = UNIFIED.state

    df = df.rename(columns=rename_map).copy()
    df[UNIFIED.retailer] = schema.name

    # Convert integer dates
    if schema.date_is_integer and UNIFIED.date in df.columns:
        df[UNIFIED.date] = convert_int_date(df[UNIFIED.date])

    return df
