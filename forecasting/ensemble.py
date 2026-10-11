"""
forecasting/ensemble.py

Meta-model: Ridge regression stacking LightGBM + LSTM outputs. Train on
validation-fold predictions (not train-fold) to avoid leakage.

Also exposes the tool-contract function:

    def forecast_demand(sku_id: str, horizon_days: int) -> ForecastOutput

as required by the Forecasting Engine spec.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from . import features as feat
from .lgbm_model import LGBMQuantileForecaster
from .lstm_model import LSTMForecasterWrapper, LSTMTrainConfig


@dataclass
class ForecastOutput:
    point_forecast: list[float]
    lower_bound: list[float]
    upper_bound: list[float]
    model_confidence: float

    def __post_init__(self):
        n = len(self.point_forecast)
        if not (len(self.lower_bound) == len(self.upper_bound) == n):
            raise ValueError("point_forecast, lower_bound, upper_bound must be equal length")
        if not (0.0 <= self.model_confidence <= 1.0):
            raise ValueError("model_confidence must be in [0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)


def _mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """Guarded MAPE — eps avoids divide-by-zero on days with 0 units sold."""
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _wape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """Weighted absolute percentage error — robust to low/zero demand."""
    return float(np.sum(np.abs(y_true - y_pred)) / max(np.sum(np.abs(y_true)), eps))


def _bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean signed error: positive = over-forecast, negative = under-forecast."""
    return float(np.mean(y_pred - y_true))


class DemandEnsemble:
    """
    Fits LightGBM (quantile x3) + LSTM on a single SKU's history, then
    stacks their point predictions with a Ridge meta-model trained on
    OUT-OF-FOLD (validation-fold) predictions from a TimeSeriesSplit,
    never on train-fold predictions, to avoid leakage.
    """

    def __init__(self, n_splits: int = 4, lstm_config: LSTMTrainConfig | None = None):
        self.n_splits = n_splits
        self.lgbm = LGBMQuantileForecaster()
        self.lstm_config = lstm_config or LSTMTrainConfig()
        self.lstm: LSTMForecasterWrapper | None = None
        self.meta = Ridge(alpha=1.0)
        self.feature_cols_: list[str] | None = None
        self.last_mape_: float | None = None

    def fit(self, sku_df: pd.DataFrame, target_col: str = "Units Sold"):
        fdf = feat.build_feature_frame(sku_df, target_col=target_col)
        self.feature_cols_ = feat.get_feature_columns(fdf, target_col=target_col)
        fdf = fdf.dropna(subset=self.feature_cols_ + [target_col]).reset_index(drop=True)

        min_hist = feat.min_history_required()
        if len(fdf) < min_hist + 20:
            raise ValueError(
                f"SKU has only {len(fdf)} usable rows after feature engineering; "
                f"need at least {min_hist + 20} for a reliable train/val split."
            )

        X = fdf[self.feature_cols_].to_numpy(dtype=float)
        y = fdf[target_col].to_numpy(dtype=float)

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        oof_lgbm = np.full(len(y), np.nan)
        oof_lstm = np.full(len(y), np.nan)
        fold_mapes = []

        splits = list(tscv.split(X))
        for train_idx, val_idx in splits:
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            fold_lgbm = LGBMQuantileForecaster()
            fold_lgbm.fit(X_tr, y_tr)
            lgbm_pred = fold_lgbm.predict(X_val)["median"]
            oof_lgbm[val_idx] = lgbm_pred

            try:
                fold_lstm = LSTMForecasterWrapper(n_features=X.shape[1], config=self.lstm_config)
                fold_lstm.fit(X_tr, y_tr, X_val, y_val)
                seq = self.lstm_config.seq_len
                # predict one-step-ahead for each val row that has enough
                # preceding history within this fold's val slice
                lstm_val_pred = np.full(len(y_val), np.nan)
                X_val_norm = fold_lstm._normalize(X_val, fit=False)
                import torch
                fold_lstm.model.eval()
                with torch.no_grad():
                    for i in range(seq, len(X_val)):
                        window = X_val_norm[i - seq:i]
                        x = torch.from_numpy(window[None, :, :].astype(np.float32))
                        lstm_val_pred[i] = fold_lstm.model(x).item()
                oof_lstm[val_idx] = lstm_val_pred
            except ValueError:
                pass  # fold too short for LSTM sequence length; leave NaN

            fold_mapes.append(_mape(y_val, lgbm_pred))

        # Meta-model trained only where both base models produced a
        # prediction (out-of-fold, never on train-fold rows).
        mask = ~np.isnan(oof_lgbm) & ~np.isnan(oof_lstm)
        if mask.sum() < 10:
            # not enough overlap to stack LSTM; fall back to LightGBM-only
            self.meta.fit(oof_lgbm[~np.isnan(oof_lgbm)].reshape(-1, 1),
                          y[~np.isnan(oof_lgbm)])
            self._lstm_available = False
        else:
            stack_X = np.column_stack([oof_lgbm[mask], oof_lstm[mask]])
            self.meta.fit(stack_X, y[mask])
            self._lstm_available = True

        self.last_mape_ = float(np.mean(fold_mapes))

        # Final fit on ALL data for deployment-time forecasting
        self.lgbm.fit(X, y)
        try:
            n_val = max(int(0.15 * len(X)), feat.min_history_required() + 5)
            self.lstm = LSTMForecasterWrapper(n_features=X.shape[1], config=self.lstm_config)
            self.lstm.fit(X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:])
        except ValueError:
            self.lstm = None
            self._lstm_available = False

        self._last_X = X
        self._last_y = y
        return self

    def predict(self, horizon_days: int) -> ForecastOutput:
        if self.feature_cols_ is None:
            raise RuntimeError("Call .fit() before .predict().")

        seq = self.lstm_config.seq_len
        last_features = self._last_X[-max(seq, 1):]

        lgbm_last = self.lgbm.predict(self._last_X[-1:].repeat(horizon_days, axis=0))
        lgbm_point = lgbm_last["median"]
        lgbm_lower = lgbm_last["lower"]
        lgbm_upper = lgbm_last["upper"]

        if self._lstm_available and self.lstm is not None and len(last_features) >= seq:
            lstm_point = self.lstm.predict_horizon(last_features[-seq:], horizon_days)
            stack_X = np.column_stack([lgbm_point, lstm_point])
            point = self.meta.predict(stack_X)
        else:
            point = self.meta.predict(lgbm_point.reshape(-1, 1))

        # widen/shift the LightGBM quantile band by the same delta the
        # meta-model applied to the median, so the interval still brackets
        # the ensembled point forecast
        delta = point - lgbm_point
        lower = lgbm_lower + delta
        upper = lgbm_upper + delta
        lower = np.minimum(lower, point)
        upper = np.maximum(upper, point)

        confidence = float(np.clip(1.0 - (self.last_mape_ or 0.5), 0.0, 1.0))

        return ForecastOutput(
            point_forecast=point.tolist(),
            lower_bound=lower.tolist(),
            upper_bound=upper.tolist(),
            model_confidence=confidence,
        )


# ---------------------------------------------------------------------
# Tool contract
# ---------------------------------------------------------------------

_MODEL_CACHE: dict[str, DemandEnsemble] = {}
_DATA_SOURCE: pd.DataFrame | None = None


def set_data_source(df: pd.DataFrame) -> None:
    """
    Register the working dataset (already loaded via forecasting.data)
    so forecast_demand() can look up a SKU's history by id.
    """
    global _DATA_SOURCE
    _DATA_SOURCE = df


def forecast_demand(sku_id: str, horizon_days: int) -> ForecastOutput:
    """
    Exposed tool function.

    sku_id: composite "{Store ID}::{Product ID}" (see forecasting.data.attach_sku_id)
    horizon_days: number of future days to forecast (e.g. 7, 14, 30)

    Returns a ForecastOutput with point_forecast / lower_bound / upper_bound
    each of length horizon_days, plus model_confidence in [0, 1].
    """
    if _DATA_SOURCE is None:
        raise RuntimeError("No data source registered — call set_data_source(df) first.")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be a positive integer.")

    sku_df = _DATA_SOURCE[_DATA_SOURCE["sku_id"] == sku_id].copy()
    if sku_df.empty:
        raise ValueError(f"Unknown sku_id: {sku_id!r}")

    if sku_id not in _MODEL_CACHE:
        model = DemandEnsemble()
        model.fit(sku_df)
        _MODEL_CACHE[sku_id] = model

    return _MODEL_CACHE[sku_id].predict(horizon_days)
