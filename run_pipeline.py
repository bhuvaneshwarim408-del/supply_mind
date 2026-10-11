"""
run_pipeline.py -- runs the full real pipeline: real CSV -> Sam's real
forecast_demand() -> your real optimize_inventory() -> real numbers.
"""
import time
from forecasting.data import load_csv
from forecasting import forecast_demand, set_data_source
from shared.schemas import ForecastOutput as SchemaForecastOutput, OptimizationConstraints
from optimization import optimize_inventory

print("Loading real retail_store_inventory.csv...")
df = load_csv("data/retail_store_inventory.csv")
print(f"  loaded {len(df)} rows")
set_data_source(df)

sku_id = "S001::P0001"
print(f"Training real ensemble for {sku_id} (this takes ~1 minute)...")
t0 = time.time()
raw_forecast = forecast_demand(sku_id, horizon_days=14)
print(f"  done in {time.time()-t0:.1f}s")
print(f"  point_forecast (first 5 days): {[round(x,2) for x in raw_forecast.point_forecast[:5]]}")
print(f"  model_confidence: {raw_forecast.model_confidence:.3f}")

forecast = SchemaForecastOutput(
    sku_id=sku_id, horizon_days=14,
    point_forecast=raw_forecast.point_forecast,
    lower_bound=raw_forecast.lower_bound,
    upper_bound=raw_forecast.upper_bound,
    model_confidence=raw_forecast.model_confidence,
)
constraints = OptimizationConstraints(
    warehouse_capacity=500, budget=5000, lead_time_days=7,
    unit_holding_cost=2.0, unit_stockout_penalty=10.0, min_order_qty=20,
)

result = optimize_inventory(forecast, constraints)
print()
print("=== OptimizationOutput ===")
print(f"  order_quantity:      {result.order_quantity}")
print(f"  safety_stock:        {result.safety_stock}")
print(f"  reorder_point:       {result.reorder_point}")
print(f"  total_expected_cost: {result.total_expected_cost}")
print(f"  solver_status:       {result.solver_status}")