"""
forecasting/features.py

Lag features (t-1, t-7, t-14, t-28); rolling mean/std over 7-day and
28-day windows; calendar dummies (day-of-week, month, holiday flag via
pandas.tseries.holiday).

All lag/rolling ops are computed PER SKU (Store ID + Product ID) so
history never leaks across series.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from .data import SKU_KEY_COLUMNS

LAGS = (1, 7, 14, 28)
ROLLING_WINDOWS = (7, 28)

# Exogenous raw columns that are safe to use as-is (no leakage: known at
# decision time, or forward-fillable business inputs).
CATEGORICAL_COLUMNS = ["Category", "Region", "Weather Condition", "Seasonality"]
NUMERIC_EXOG_COLUMNS = [
    "Price", "Discount", "Competitor Pricing", "Holiday/Promotion",
]

_holiday_cal = USFederalHolidayCalendar()


def add_calendar_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    df = df.copy()
    dt = df[date_col]
    df["dow"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    lo, hi = dt.min(), dt.max()
    holidays = _holiday_cal.holidays(start=lo, end=hi)
    df["is_holiday"] = dt.isin(holidays).astype(int)

    # one-hot the low-cardinality calendar dummies
    df = pd.get_dummies(df, columns=["dow", "month"], prefix=["dow", "month"])
    return df


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    target_col: str = "Units Sold",
    lags: Iterable[int] = LAGS,
    windows: Iterable[int] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """
    Must be called on data already sorted by SKU_KEY_COLUMNS + Date.
    Adds lag_{k} and roll_mean_{w}/roll_std_{w}, grouped per SKU so no
    cross-series leakage occurs. Rolling stats are shifted by 1 day so
    day t's features never see day t's own target.
    """
    df = df.copy()
    grp = df.groupby(SKU_KEY_COLUMNS, sort=False)[target_col]

    for k in lags:
        df[f"lag_{k}"] = grp.shift(k)

    shifted = grp.shift(1)  # exclude current day from rolling window
    shifted_grp = shifted.groupby(df[SKU_KEY_COLUMNS[0]].astype(str) + "::" +
                                   df[SKU_KEY_COLUMNS[1]].astype(str))
    for w in windows:
        df[f"roll_mean_{w}"] = shifted_grp.transform(
            lambda s: s.rolling(window=w, min_periods=max(2, w // 4)).mean()
        )
        df[f"roll_std_{w}"] = shifted_grp.transform(
            lambda s: s.rolling(window=w, min_periods=max(2, w // 4)).std()
        )
    return df


def encode_categoricals(df: pd.DataFrame, columns=CATEGORICAL_COLUMNS) -> pd.DataFrame:
    df = df.copy()
    present = [c for c in columns if c in df.columns]
    df = pd.get_dummies(df, columns=present, prefix=present)
    return df


def build_feature_frame(
    df: pd.DataFrame,
    target_col: str = "Units Sold",
) -> pd.DataFrame:
    """
    Full feature pipeline: calendar dummies + lag/rolling + categorical
    encoding. Returns a frame with NaNs still present in early rows
    (insufficient history) — caller decides how to handle (drop or
    route to a fallback model).
    """
    out = add_calendar_features(df)
    out = add_lag_and_rolling_features(out, target_col=target_col)
    out = encode_categoricals(out)
    return out


def get_feature_columns(df: pd.DataFrame, target_col: str = "Units Sold") -> list[str]:
    """
    Everything usable as a model input: excludes identifiers, the raw
    Date, the target itself, and known-leaky/benchmark columns.
    """
    from .data import LEAKY_OR_TARGET_COLUMNS

    exclude = set(SKU_KEY_COLUMNS) | LEAKY_OR_TARGET_COLUMNS | {
        "Date", "sku_id", target_col,
    }
    return [c for c in df.columns if c not in exclude and df[c].dtype != "O"]


def min_history_required() -> int:
    """LSTM sequence length + longest lag/rolling window."""
    return max(LAGS) + max(ROLLING_WINDOWS)
