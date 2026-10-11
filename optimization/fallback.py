"""
optimization/fallback.py
Owner: Bhuvaneshwari M (1VJ23CS009)
Branch: feature/optimization-fallback

Plain cvxpy.Problem().solve() -- no CvxpyLayer, no gradients. This must keep
working at ALL times as insurance in case the differentiable layer breaks close
to the deadline. solver_status will read "fallback" when this path is used
instead of the differentiable layer.
"""

from shared.schemas import OptimizationConstraints, OptimizationOutput
from optimization.core import build_problem, compute_reorder_point


def solve_fallback(demand_mean: float, demand_std: float, constraints: OptimizationConstraints) -> OptimizationOutput:
    """Plain solve, no autograd. Always keep this working as a safety net."""
    problem, params, variables = build_problem()

    params["demand_mean"].value = demand_mean
    params["unit_holding_cost"].value = constraints.unit_holding_cost
    params["unit_stockout_penalty"].value = constraints.unit_stockout_penalty
    # TODO: OptimizationConstraints has no dedicated per-unit purchase cost field
    # yet -- using unit_holding_cost as a stand-in until shared/schemas.py grows
    # a real `unit_cost` field (needs sign-off from all 4 members).
    params["unit_cost"].value = constraints.unit_holding_cost
    params["budget"].value = constraints.budget
    params["capacity"].value = constraints.warehouse_capacity

    try:
        problem.solve()
        status = "optimal" if problem.status == "optimal" else "infeasible"
    except Exception:
        status = "infeasible"

    q_val = variables["order_quantity"].value
    s_val = variables["safety_stock"].value

    if q_val is None or status == "infeasible":
        return OptimizationOutput(
            order_quantity=0.0,
            safety_stock=0.0,
            reorder_point=0.0,
            total_expected_cost=0.0,
            solver_status="infeasible",
        )

    q_val = max(float(q_val), constraints.min_order_qty)  # enforce supplier MOQ post-hoc
    s_val = float(s_val)

    # r is plain parameter arithmetic, computed outside the CVXPY problem --
    # see optimization/core.py module docstring for why.
    r_val = compute_reorder_point(demand_mean, demand_std, constraints.lead_time_days)

    return OptimizationOutput(
        order_quantity=round(q_val, 2),
        safety_stock=round(s_val, 2),
        reorder_point=round(float(r_val), 2),
        total_expected_cost=round(float(problem.value), 2) if problem.value is not None else 0.0,
        solver_status="fallback",
    )
