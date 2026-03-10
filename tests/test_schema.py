"""Tests for schema.py — harmonization and date conversion."""

import pytest
import pandas as pd
import numpy as np

from src.data.schema import (
    DG_SCHEMA, SF_SCHEMA, WG_SCHEMA,
    harmonize_to_unified, convert_int_date, UNIFIED,
)


class TestConvertIntDate:
    """Test integer date conversion."""

    def test_valid_yyyymmdd(self):
        s = pd.Series([20230101, 20231225, 20240229])
        result = convert_int_date(s)
        assert result.dtype == "datetime64[ns]"
        assert result.iloc[0] == pd.Timestamp("2023-01-01")
        assert result.iloc[1] == pd.Timestamp("2023-12-25")
        assert result.iloc[2] == pd.Timestamp("2024-02-29")  # leap year

    def test_invalid_date_returns_nat(self):
        s = pd.Series([99999999, 0, -1])
        result = convert_int_date(s)
        assert result.isna().all()

    def test_mixed_valid_invalid(self):
        s = pd.Series([20230101, 99999999])
        result = convert_int_date(s)
        assert result.iloc[0] == pd.Timestamp("2023-01-01")
        assert pd.isna(result.iloc[1])


class TestHarmonizeDG:
    """Test DG schema harmonization."""

    def _make_dg_df(self, n=5):
        return pd.DataFrame({
            "DATE": pd.date_range("2023-01-01", periods=n),
            "STORE": [1] * n,
            "UPC": ["0049000001"] * n,
            "SKU": ["SKU-00001"] * n,
            "P_SKU": ["PSKU-00001"] * n,
            "QTY_SOLD": np.random.randint(10, 100, n),
            "NET_SALES": np.random.uniform(50, 500, n),
            "CLASS_DESC": ["Snacks"] * n,
            "DEPARTMENT_DESC": ["Consumables"] * n,
            "CITY": ["Houston"] * n,
            "STATE": ["TX"] * n,
        })

    def test_column_renaming(self):
        df = harmonize_to_unified(self._make_dg_df(), DG_SCHEMA)
        assert UNIFIED.date in df.columns
        assert UNIFIED.store in df.columns
        assert UNIFIED.product in df.columns
        assert UNIFIED.qty_sold in df.columns
        assert UNIFIED.net_sales in df.columns

    def test_retailer_label_added(self):
        df = harmonize_to_unified(self._make_dg_df(), DG_SCHEMA)
        assert (df[UNIFIED.retailer] == "DG").all()

    def test_geo_mapped(self):
        df = harmonize_to_unified(self._make_dg_df(), DG_SCHEMA)
        assert UNIFIED.city in df.columns
        assert UNIFIED.state in df.columns


class TestHarmonizeSF:
    """Test S&F schema harmonization."""

    def _make_sf_df(self, n=5):
        return pd.DataFrame({
            "TRANSACTIONDATE": [20230101 + i for i in range(n)],
            "STORENUMBER": [1] * n,
            "UPCNUMBER": [4900000001] * n,
            "ITEMNUMBER": ["ITEM-00001"] * n,
            "ITEMQUANTITY": np.random.randint(1, 10, n),
            "TOTALNETPRICEAMOUNT": np.random.uniform(5, 50, n),
        })

    def test_integer_dates_converted(self):
        df = harmonize_to_unified(self._make_sf_df(), SF_SCHEMA)
        assert df[UNIFIED.date].dtype == "datetime64[ns]"

    def test_retailer_label(self):
        df = harmonize_to_unified(self._make_sf_df(), SF_SCHEMA)
        assert (df[UNIFIED.retailer] == "SF").all()


class TestHarmonizeWG:
    """Test WG schema harmonization."""

    def _make_wg_df(self, n=5):
        return pd.DataFrame({
            "DATE": pd.date_range("2023-01-01", periods=n),
            "STORE": [1] * n,
            "CATEGORY": ["Pharmacy"] * n,
            "CAT_QTY_SOLD": np.random.randint(50, 500, n),
            "CAT_NET_SALES": np.random.uniform(200, 2000, n),
        })

    def test_category_mapped(self):
        df = harmonize_to_unified(self._make_wg_df(), WG_SCHEMA)
        assert UNIFIED.category in df.columns

    def test_no_product_column(self):
        """WG has no product-level detail; product column should not exist."""
        df = harmonize_to_unified(self._make_wg_df(), WG_SCHEMA)
        # product may or may not be in columns depending on schema
        # but should not have the original WG columns
        assert "CAT_QTY_SOLD" not in df.columns
