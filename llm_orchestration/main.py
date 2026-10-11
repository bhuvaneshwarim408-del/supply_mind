"""
main.py
-------
Public entry point for the LLM orchestration module.

Ruchitha's Streamlit UI should only ever need to call generate_explanation()
- everything else in this package is an internal implementation detail.

Expected usage (matches the shape of Sam's forecast_demand() and
Bhuvii's optimize_inventory()):

    from llm_orchestration import generate_explanation

    result = generate_explanation(sku, forecast_output, optimization_output)
    print(result["explanation"])
"""

from typing import Dict, Any

from shared.schemas import ForecastOutput, OptimizationOutput
from .explainer import explain_decision


def generate_explanation(sku: str, forecast: ForecastOutput,
                          optimization: OptimizationOutput) -> Dict[str, Any]:
    """
    Generate a plain-language, RAG-grounded explanation of an inventory
    decision for a single SKU.

    Parameters
    ----------
    sku : str
        SKU identifier, e.g. "S001::P0001"
    forecast : shared.schemas.ForecastOutput
        Output of Sam's forecast_demand(sku, horizon), wrapped in the
        team's shared contract (sku_id, horizon_days, point_forecast,
        lower_bound, upper_bound, model_confidence).
    optimization : shared.schemas.OptimizationOutput
        Output of Bhuvii's optimize_inventory(...) (order_quantity,
        safety_stock, reorder_point, total_expected_cost, solver_status).

    Returns
    -------
    dict:
        {
            "sku": str,
            "explanation": str,
            "sources": list[str]
        }
    """
    return explain_decision(sku, forecast, optimization)


if __name__ == "__main__":
    # Quick manual sanity check - run `python -m llm_orchestration.main`
    sample_forecast = ForecastOutput(
        sku_id="S001::P0001", horizon_days=14,
        point_forecast=[45.2] * 14, lower_bound=[36.0] * 14,
        upper_bound=[54.0] * 14, model_confidence=0.8,
    )
    sample_optimization = OptimizationOutput(
        order_quantity=320.0, safety_stock=85.0, reorder_point=210.0,
        total_expected_cost=1450.75, solver_status="optimal",
    )
    result = generate_explanation("S001::P0001", sample_forecast, sample_optimization)
    print(result["explanation"])
    print("\nSources used:")
    for s in result["sources"]:
        print(" -", s)
