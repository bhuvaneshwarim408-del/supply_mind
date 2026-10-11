"""
Data loading utilities.

Expected raw schema (as supplied):
Date, Store ID, Product ID, Category, Region, Inventory Level, Units Sold,
Units Ordered, Demand Forecast, Price, Discount, Weather Condition,
Holiday/Promotion, Competitor Pricing, Seasonality

Note: 'Demand Forecast' is treated as an EXTERNAL BENCHMARK column (the
retailer's own existing forecast), not a training feature — using it as
an input would leak future-looking information. It is kept aside so
docs/forecast_eval.md can optionally compare our model against it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "Date", "Store ID", "Product ID", "Category", "Region",
    "Inventory Level", "Units Sold", "Units Ordered", "Demand Forecast",
    "Price", "Discount", "Weather Condition", "Holiday/Promotion",
    "Competitor Pricing", "Seasonality",
]

# Columns that must NOT be used as model inputs (leakage / target / benchmark)
LEAKY_OR_TARGET_COLUMNS = {"Units Sold", "Demand Forecast"}

SKU_KEY_COLUMNS = ["Store ID", "Product ID"]


def load_csv(path: str) -> pd.DataFrame:
    """Load the raw CSV, parse dates, sort by SKU key + Date, build sku_id."""
    df = pd.read_csv(path)
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing expected columns: {missing}")
    df["Date"] = pd.to_datetime(df["Date"])
    df = attach_sku_id(df)
    df = df.sort_values(SKU_KEY_COLUMNS + ["Date"]).reset_index(drop=True)
    return df


def attach_sku_id(df: pd.DataFrame) -> pd.DataFrame:
    """Composite key: a Product ID alone may repeat across stores."""
    df = df.copy()
    df["sku_id"] = (
        df["Store ID"].astype(str) + "::" + df["Product ID"].astype(str)
    )
    return df


def generate_synthetic(
    n_days: int = 400,
    stores=("S001", "S002"),
    products=("P001", "P002", "P003"),
    seed: int = 42,
) -> pd.DataFrame:
    """
    Synthetic dataset matching the raw schema exactly, for local testing
    when no real CSV is available yet. Swap this out for load_csv() once
    you have the real file.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    categories = {"P001": "Beverages", "P002": "Snacks", "P003": "Dairy"}
    regions = {"S001": "South", "S002": "West"}
    weather_options = ["Sunny", "Rainy", "Cloudy", "Stormy"]

    for store in stores:
        for product in products:
            base_level = rng.uniform(30, 80)
            trend = rng.uniform(-0.01, 0.02)
            for i, d in enumerate(dates):
                dow = d.dayofweek
                weekly = 1.0 + 0.25 * np.sin(2 * np.pi * dow / 7)
                seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * d.dayofyear / 365)
                promo = int(rng.random() < 0.08)
                price = round(rng.uniform(50, 150), 2)
                discount = round(rng.uniform(0, 20) if promo else 0.0, 2)
                competitor_price = round(price * rng.uniform(0.9, 1.1), 2)
                noise = rng.normal(0, 4)
                units_sold = max(
                    0,
                    (base_level + trend * i) * weekly * seasonal
                    + (10 if promo else 0)
                    - 0.15 * discount
                    + noise,
                )
                units_sold = round(units_sold)
                inventory_level = max(0, round(units_sold * rng.uniform(1.2, 2.5)))
                units_ordered = max(0, round(units_sold * rng.uniform(0.9, 1.3)))
                naive_forecast = round(units_sold * rng.uniform(0.85, 1.15))
                weather = rng.choice(weather_options)
                season_label = ["Winter", "Spring", "Summer", "Fall"][
                    (d.month % 12) // 3
                ]
                rows.append(
                    {
                        "Date": d,
                        "Store ID": store,
                        "Product ID": product,
                        "Category": categories[product],
                        "Region": regions[store],
                        "Inventory Level": inventory_level,
                        "Units Sold": units_sold,
                        "Units Ordered": units_ordered,
                        "Demand Forecast": naive_forecast,
                        "Price": price,
                        "Discount": discount,
                        "Weather Condition": weather,
                        "Holiday/Promotion": promo,
                        "Competitor Pricing": competitor_price,
                        "Seasonality": season_label,
                    }
                )
    df = pd.DataFrame(rows)
    df = attach_sku_id(df)
    df = df.sort_values(SKU_KEY_COLUMNS + ["Date"]).reset_index(drop=True)
    return df
