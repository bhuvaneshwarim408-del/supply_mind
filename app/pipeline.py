"""
app/pipeline.py
Owner: Ruchitha S -- integration layer

Chains the REAL modules together using the team's actual shared/schemas.py
contracts:

    forecasting.forecast_demand(sku_id, horizon_days) -> raw ensemble output
        -> wrapped into shared.schemas.ForecastOutput (see note below)
    optimization.optimize_inventory(forecast, constraints) -> OptimizationOutput
    llm_orchestration.generate_explanation(sku, forecast, optimization) -> dict

IMPORTANT NOTE on forecasting's ForecastOutput:
forecasting/ensemble.py defines its OWN small dataclass called ForecastOutput
(point_forecast, lower_bound, upper_bound, model_confidence -- no sku_id).
This is DIFFERENT from shared.schemas.ForecastOutput (which adds sku_id and
horizon_days). run_pipeline.py at the project root already handles this by
manually re-wrapping the raw ensemble output into the shared schema -- this
file does the same thing, so app/main.py (the Streamlit UI) gets a single
consistent shared.schemas.ForecastOutput object throughout.

This mismatch is worth raising with Sam: ideally forecasting/ensemble.py's
forecast_demand() would just return shared.schemas.ForecastOutput directly
so this re-wrapping step isn't needed.

NOTE: llm_orchestration.generate_explanation() now accepts shared.schemas
objects directly (fixed - it used to expect differently-named dict keys,
which required an adapter here. That adapter has been removed).
"""

from __future__ import annotations

from shared.schemas import ForecastOutput, OptimizationConstraints, OptimizationOutput
from forecasting import forecast_demand, set_data_source
from forecasting.data import load_csv
from optimization import optimize_inventory

try:
    from llm_orchestration import generate_explanation
    EXPLANATION_MODULE_AVAILABLE = True
except ImportError:
    EXPLANATION_MODULE_AVAILABLE = False


def _call_explanation(sku_id: str, forecast: ForecastOutput, optimization: OptimizationOutput):
    """Thin wrapper - just adds try/except so a failed explanation call
    (e.g. missing ANTHROPIC_API_KEY) never breaks the rest of the pipeline."""
    if not EXPLANATION_MODULE_AVAILABLE:
        return None
    try:
        return generate_explanation(sku_id, forecast, optimization)
    except Exception as e:
        return {"sku": sku_id, "explanation": None, "sources": [], "error": str(e)}


def run_pipeline(
    sku_id: str,
    horizon_days: int,
    constraints: OptimizationConstraints,
    csv_path: str = "data/retail_store_inventory.csv",
) -> dict:
    """
    The single function app/main.py (Streamlit UI) calls.

    Returns a dict:
        {
            "forecast": ForecastOutput | None,
            "optimization": OptimizationOutput | None,
            "explanation": dict | None,
            "error": str | None,
        }
    Never raises - always returns this shape so the UI can render a clean
    error message instead of crashing.
    """
    try:
        df = load_csv(csv_path)
        set_data_source(df)
        raw_forecast = forecast_demand(sku_id, horizon_days)
        # Re-wrap forecasting's internal dataclass into the shared contract -
        # see module docstring above for why this step exists.
        forecast = ForecastOutput(
            sku_id=sku_id,
            horizon_days=horizon_days,
            point_forecast=raw_forecast.point_forecast,
            lower_bound=raw_forecast.lower_bound,
            upper_bound=raw_forecast.upper_bound,
            model_confidence=raw_forecast.model_confidence,
        )
    except Exception as e:
        return {"forecast": None, "optimization": None, "explanation": None,
                "error": f"Forecasting stage failed: {e}"}

    try:
        optimization = optimize_inventory(forecast, constraints)
    except Exception as e:
        return {"forecast": forecast, "optimization": None, "explanation": None,
                "error": f"Optimization stage failed: {e}"}

    explanation = _call_explanation(sku_id, forecast, optimization)

    return {"forecast": forecast, "optimization": optimization, "explanation": explanation, "error": None}


if __name__ == "__main__":
    # Manual check: run `python -m app.pipeline` from the project root.
    result = run_pipeline(
        sku_id="S001::P0001",
        horizon_days=14,
        constraints=OptimizationConstraints(
            warehouse_capacity=500, budget=5000, lead_time_days=7,
            unit_holding_cost=2.0, unit_stockout_penalty=10.0, min_order_qty=20,
        ),
    )
    print("Error:", result["error"])
    if result["optimization"]:
        print("Optimization:", result["optimization"].model_dump())
    if result["explanation"]:
        print("Explanation:", result["explanation"])
