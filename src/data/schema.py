"""
Column role mapping.

When the real schema is provided, populate COLUMN_ROLES with
the actual column names mapped to their semantic roles.
All downstream pipeline logic should reference these roles,
not hard-coded column names.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ColumnRoles:
    """Semantic mapping of dataset columns to their roles."""

    # Required
    time_key: str = "date"
    retailer_key: str = "retailer_id"
    object_key: str = "object_id"
    primary_target: str = "units_sold"

    # Optional hierarchy
    category: Optional[str] = "category"
    brand: Optional[str] = "brand"
    region: Optional[str] = "region"
    subcategory: Optional[str] = None
    channel: Optional[str] = None
    market: Optional[str] = None

    # Optional driver features
    price: Optional[str] = "price"
    promo_flag: Optional[str] = "promo_flag"
    promo_discount: Optional[str] = None
    inventory: Optional[str] = None
    distribution: Optional[str] = None

    # Optional event/external features
    holiday_flag: Optional[str] = "holiday_flag"
    event_flag: Optional[str] = None
    weather_index: Optional[str] = None

    # Secondary targets
    secondary_targets: list[str] = field(default_factory=lambda: ["revenue", "margin"])

    def entity_keys(self) -> list[str]:
        """All key columns that define a unique time series."""
        return [self.retailer_key, self.object_key]

    def hierarchy_fields(self) -> list[str]:
        """Non-null hierarchy columns."""
        candidates = [self.category, self.brand, self.region,
                      self.subcategory, self.channel, self.market]
        return [c for c in candidates if c is not None]

    def driver_features(self) -> list[str]:
        """Non-null driver feature columns."""
        candidates = [self.price, self.promo_flag, self.promo_discount,
                      self.inventory, self.distribution]
        return [c for c in candidates if c is not None]

    def event_features(self) -> list[str]:
        """Non-null event/external feature columns."""
        candidates = [self.holiday_flag, self.event_flag, self.weather_index]
        return [c for c in candidates if c is not None]


# Singleton instance — update this when real schema arrives
SCHEMA = ColumnRoles()
