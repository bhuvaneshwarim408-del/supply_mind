"""
forecasting/main.py
Owner: D Sam Goldwin (1VJ23CS013)

Exposes the single tool-contract function forecast_demand(), matching the signature
that orchestration/tools.py and app/main.py call against. This is what Harshitha's
agent_loop.py and Ruchitha's Streamlit app import once USE_STUBS = False.
"""

from shared.schemas import ForecastOutput

# TODO: load trained lgbm models, lstm model, and meta-model here (e.g. from data/models/*.pkl)
# from forecasting.lgbm_model import predict_quantiles
# from forecasting.lstm_model import forecast_horizon
# from forecasting.ensemble import ensemble_predict


def forecast_demand(sku_id: str, horizon_days: int) -> ForecastOutput:
    """
    Tool contract -- signature must not change without updating shared/schemas.py
    and notifying all 4 members.

    TODO: replace this body with the real pipeline:
      1. Pull recent history for sku_id from data/
      2. Build features (forecasting.features.build_feature_matrix)
      3. Run LightGBM quantile models -> lower/point/upper
      4. Run LSTM -> point forecast
      5. Combine via ensemble.ensemble_predict -> final point_forecast
      6. Compute model_confidence = 1 - normalized MAPE from the latest backtest
         (see docs/forecast_eval.md)
    """
    raise NotImplementedError(
        "forecast_demand: wire up the real pipeline. "
        "Until then, use shared.stubs.forecast_demand_stub for integration testing."
    )
