"""
smoke_test.py
-------------
Quick end-to-end check with fabricated sample data (no real forecast/
optimization outputs needed). Run this first, before wiring up Sam's and
Bhuvii's real outputs, to confirm the module doesn't crash.

Usage:
    python smoke_test.py
"""

from llm_orchestration import generate_explanation
from shared.schemas import ForecastOutput, OptimizationOutput


def main():
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

    assert "explanation" in result
    assert "sources" in result
    assert isinstance(result["sources"], list)

    print("SMOKE TEST PASSED\n")
    print("SKU:", result["sku"])
    print("Explanation:\n", result["explanation"])
    print("\nSources:")
    for s in result["sources"]:
        print(" -", s)


if __name__ == "__main__":
    main()
