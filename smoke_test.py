"""
smoke_test.py
Run this first, at the project root: python smoke_test.py

Just confirms every module imports cleanly and the optimization problem is
DPP-compliant. Does not train anything or touch real data yet.
"""

print("Checking imports...")
from optimization.core import build_problem, build_layer
from optimization import optimize_inventory
from shared.schemas import ForecastOutput, OptimizationConstraints
print("  optimization + shared imports OK")

from forecasting import forecast_demand, set_data_source
from forecasting.data import load_csv
print("  forecasting imports OK")

print()
print("Checking DPP compliance of your CVXPY problem...")
problem, params, variables = build_problem()
assert problem.is_dcp(dpp=True), "DPP check FAILED"
print("  is_dcp(dpp=True):", problem.is_dcp(dpp=True), " <- should be True")

print()
print("ALL GOOD. You're ready for the next step.")