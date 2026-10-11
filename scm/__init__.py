"""Decision-making core: data, forecasters, inventory simulator, policies, RL and backtests.

Runs on numpy/pandas/cvxpy only (no torch), so every experiment reproduces on a laptop in minutes.
The heavier LightGBM+LSTM ensemble in forecasting/ is used for live single-SKU forecasts in the web app.
"""
