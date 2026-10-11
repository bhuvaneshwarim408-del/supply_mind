"""
shared/stubs.py

Fake implementations of every exposed tool function, matching the real signatures
exactly. This is what lets all four branches build and unit-test in parallel in
Week 1-2 before any real model exists.

Each real module (forecasting/, optimization/, orchestration/) should eventually
expose a function with an IDENTICAL signature to the one here. app/main.py swaps
between stub and real implementation with a single flag: USE_STUBS.
"""

import random
from datetime import datetime, timezone

from shared.schemas import Alert, ForecastOutput, OptimizationConstraints, OptimizationOutput


def forecast_demand_stub(sku_id: str, horizon_days: int) -> ForecastOutput:
    """Fake version of forecasting.forecast_demand(). Deterministic-ish fake curve."""
    base = 50 + (hash(sku_id) % 30)
    point = [round(base + 5 * random.uniform(-1, 1), 2) for _ in range(horizon_days)]
    lower = [round(p * 0.8, 2) for p in point]
    upper = [round(p * 1.2, 2) for p in point]
    return ForecastOutput(
        sku_id=sku_id,
        horizon_days=horizon_days,
        point_forecast=point,
        lower_bound=lower,
        upper_bound=upper,
        model_confidence=0.75,
    )


def optimize_inventory_stub(
    forecast: ForecastOutput, constraints: OptimizationConstraints
) -> OptimizationOutput:
    """Fake version of optimization.optimize_inventory(). Simple heuristic, no solver."""
    avg_demand = sum(forecast.point_forecast) / len(forecast.point_forecast)
    order_quantity = max(constraints.min_order_qty, avg_demand * constraints.lead_time_days)
    safety_stock = avg_demand * 0.5
    reorder_point = avg_demand * constraints.lead_time_days + safety_stock
    total_cost = (
        order_quantity * constraints.unit_holding_cost
        + safety_stock * constraints.unit_holding_cost
    )
    order_quantity = min(order_quantity, constraints.warehouse_capacity)
    return OptimizationOutput(
        order_quantity=round(order_quantity, 2),
        safety_stock=round(safety_stock, 2),
        reorder_point=round(reorder_point, 2),
        total_expected_cost=round(total_cost, 2),
        solver_status="optimal",
    )


def retrieve_policy_stub(query: str) -> list[str]:
    """Fake version of orchestration.rag.retrieve_policy(). Returns canned policy text."""
    return [
        f"[stub] Policy chunk 1 relevant to '{query}': maintain minimum 2 weeks of safety stock.",
        f"[stub] Policy chunk 2 relevant to '{query}': prioritize suppliers with < 7 day lead time.",
        f"[stub] Policy chunk 3 relevant to '{query}': budget approval required above $50,000 per order.",
    ]


def make_alert_stub(sku_id: str, severity: str = "warning") -> Alert:
    """Fake version of an orchestration.automation-generated alert."""
    return Alert(
        sku_id=sku_id,
        severity=severity,  # type: ignore[arg-type]
        message=f"[stub] Days-of-cover for {sku_id} has dropped below threshold.",
        triggered_at=datetime.now(timezone.utc),
    )
