"""
optimization/main.py
Owner: Bhuvaneshwari M (1VJ23CS009)

Exposes the single tool-contract function optimize_inventory(), matching the
signature that orchestration/tools.py and app/main.py call against.
"""

from shared.schemas import ForecastOutput, OptimizationConstraints, OptimizationOutput
from optimization.core import compute_reorder_point


def _demand_stats(forecast: ForecastOutput) -> tuple[float, float]:
    """
    demand_mean <- average of forecast.point_forecast over the horizon.
    demand_std  <- derived from the average (upper_bound - lower_bound) spread,
    treating it as an 80% interval (10th/90th percentile band from the LGBM
    quantile heads -- see forecasting/lgbm_model.py) and converting to a
    standard deviation via the normal-distribution approximation
    (width_80 = 2 * 1.28 * sigma  ->  sigma = width_80 / 2.56).
    """
    demand_mean = sum(forecast.point_forecast) / len(forecast.point_forecast)
    spread = [u - l for u, l in zip(forecast.upper_bound, forecast.lower_bound)]
    avg_spread = sum(spread) / len(spread)
    demand_std = avg_spread / 2.56
    return demand_mean, demand_std


def optimize_inventory(forecast: ForecastOutput, constraints: OptimizationConstraints) -> OptimizationOutput:
    """
    Tool contract -- signature must not change without updating shared/schemas.py
    and notifying all 4 members.

    Tries the differentiable CvxpyLayer path first (optimization.core.build_layer).
    Falls back to a plain, non-differentiable solve (optimization.fallback.solve_fallback)
    if the layer isn't available (e.g. cvxpylayers/torch not installed) or the solve
    fails for any reason. solver_status on the returned OptimizationOutput reflects
    which path actually produced the result ("optimal" vs "fallback" vs "infeasible").
    """
    demand_mean, demand_std = _demand_stats(forecast)

    try:
        return _solve_with_layer(demand_mean, demand_std, constraints)
    except Exception:
        from optimization.fallback import solve_fallback
        return solve_fallback(demand_mean, demand_std, constraints)


def _solve_with_layer(
    demand_mean: float, demand_std: float, constraints: OptimizationConstraints
) -> OptimizationOutput:
    """Differentiable path: solve via CvxpyLayer, using torch tensors throughout."""
    import torch

    from optimization.core import build_layer

    layer, params, variables = build_layer()

    # Order MUST match the order build_layer() wraps parameters in (dict insertion
    # order, which matches build_problem()'s `parameters` dict).
    values = {
        "demand_mean": demand_mean,
        "unit_holding_cost": constraints.unit_holding_cost,
        "unit_stockout_penalty": constraints.unit_stockout_penalty,
        "unit_cost": constraints.unit_holding_cost,  # TODO: real unit_cost field, see fallback.py
        "budget": constraints.budget,
        "capacity": constraints.warehouse_capacity,
    }
    tensor_args = [
        torch.tensor(float(values[name]), dtype=torch.float64, requires_grad=True)
        for name in params.keys()
    ]

    q_t, s_t, shortfall_t = layer(*tensor_args)

    q_val = max(float(q_t.item()), constraints.min_order_qty)
    s_val = float(s_t.item())

    r_val = compute_reorder_point(demand_mean, demand_std, constraints.lead_time_days)

    total_cost = (
        constraints.unit_holding_cost * s_val
        + constraints.unit_holding_cost * q_val  # ordering cost stand-in, see TODO above
        + constraints.unit_stockout_penalty * max(0.0, demand_mean - (q_val + s_val))
    )

    return OptimizationOutput(
        order_quantity=round(q_val, 2),
        safety_stock=round(s_val, 2),
        reorder_point=round(float(r_val), 2),
        total_expected_cost=round(total_cost, 2),
        solver_status="optimal",
    )
