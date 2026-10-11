"""
Forecasting Engine -- Member 1 (D Sam Goldwin), branch: feature/forecasting

Exposes forecast_demand(sku_id, horizon_days) -> ForecastOutput
"""
from .ensemble import forecast_demand, set_data_source, ForecastOutput  # noqa: F401
