"""
shared/schemas.py

THE CONTRACT. Every module's input/output is one of these Pydantic models.
Changes here require sign-off from all 4 members (Sam, Bhuvaneshwari, Harshitha, Ruchitha)
per the GitHub Workflow section of the build spec.

Do not import from forecasting/, optimization/, orchestration/, or app/ in this file --
this module has zero dependencies on the others, by design, so it can be committed
in Week 1 before any model code exists.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ForecastOutput(BaseModel):
    """Returned by forecasting.forecast_demand(). Owner: Sam."""

    sku_id: str
    horizon_days: int
    point_forecast: list[float] = Field(description="Median demand estimate per day. len == horizon_days")
    lower_bound: list[float] = Field(description="10th-percentile demand estimate per day")
    upper_bound: list[float] = Field(description="90th-percentile demand estimate per day")
    model_confidence: float = Field(ge=0.0, le=1.0, description="1 - normalized MAPE on validation split")


class OptimizationConstraints(BaseModel):
    """Input to optimization.optimize_inventory(), alongside a ForecastOutput. Owner: Bhuvaneshwari."""

    warehouse_capacity: float
    budget: float
    lead_time_days: int
    unit_holding_cost: float
    unit_stockout_penalty: float
    min_order_qty: float


class OptimizationOutput(BaseModel):
    """Returned by optimization.optimize_inventory(). Owner: Bhuvaneshwari."""

    order_quantity: float
    safety_stock: float
    reorder_point: float
    total_expected_cost: float
    solver_status: Literal["optimal", "infeasible", "fallback"]


class Alert(BaseModel):
    """Produced by orchestration.automation and consumed by app.alerts_panel. Owner: Harshitha / Ruchitha."""

    sku_id: str
    severity: Literal["info", "warning", "critical"]
    message: str
    triggered_at: datetime
