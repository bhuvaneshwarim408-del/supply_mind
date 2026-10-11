"""
optimization/core.py
Owner: Bhuvaneshwari M (1VJ23CS009)

CRITICAL: Formulate the convex program by hand first and verify it on paper before
coding. A DPP (Disciplined Parametrized Programming) violation compiles WITHOUT
ERROR but silently returns zero gradients -- which would invalidate the
"differentiable" claim in the project title without anyone noticing until the
defense. Always check problem.is_dcp(dpp=True) == True before trusting this.

Design note (see docs/optimization_formulation.md for the full derivation):
The reorder point r = mu*L + z*sigma*sqrt(L) is NOT part of the CVXPY problem.
It is pure parameter arithmetic -- the solver never has a choice to make about
r, so forcing it into a cp.Constraint would multiply two parameters together
(demand_mean * lead_time_days, and demand_std * sqrt(lead_time_days)), which
violates DPP's rule that parameters may only combine AFFINELY with each other.
is_dcp(dpp=True) actually catches this (verified empirically). Instead, r is
computed directly with compute_reorder_point() below, which is ordinary
differentiable arithmetic -- autograd handles it with no CvxpyLayer needed,
since it is already a closed form in terms of the parameters.

Second DPP trap (also verified empirically): cp.pos(demand_mean - (q+s)) traps a
parameter (demand_mean) inside a nonlinear atom (pos), and unit_stockout_penalty *
that whole thing is then parameter * (nonlinear-function-of-a-parameter) --
disguised parameter*parameter, same rule as above, one level deeper. Fixed with
the standard LP reformulation: introduce an auxiliary `shortfall` variable and an
inequality constraint (shortfall >= demand_mean - (q+s)) instead of cp.pos(). At
the optimum, minimizing a positive multiple of `shortfall` pushes it down to
exactly max(0, demand_mean - (q+s)), so the reformulation is mathematically
equivalent -- but now demand_mean only ever appears affinely, alone, in a
constraint, never multiplied against another parameter.

Build order (Week 1-2):
  1. Variables:  q, s, shortfall = cp.Variable(nonneg=True) each
                 (r is NOT a solver variable -- see above)
  2. Parameters: demand_mean, unit_holding_cost, unit_stockout_penalty, unit_cost,
                 budget, capacity -- MUST be cp.Parameter, never hardcoded floats,
                 or the layer is not differentiable w.r.t. them.
  3. Objective:  minimize c_h*s + c_o*q + c_p*shortfall
  4. Constraints:
       q + s <= capacity
       q * unit_cost <= budget
       shortfall >= demand_mean - (q + s)
  5. Wrap in CvxpyLayer(problem, parameters=[...], variables=[q, s, shortfall])
     -- see build_layer()
  6. Gradient check: torch.autograd.gradcheck in tests/test_gradients.py -- this IS
     the evidence for the differentiability claim, don't skip it.
"""

import cvxpy as cp

Z_SCORE_95 = 1.65  # z for 95% service level; swap if target service level changes
REG_EPS = 1e-4  # quadratic regularization strength -- see build_problem() docstring


def build_problem() -> tuple[cp.Problem, dict[str, cp.Parameter], dict[str, cp.Variable]]:
    """
    Build the convex program for order_quantity (q) and safety_stock (s) only.
    Returns (problem, parameters_dict, variables_dict) so callers can bind values
    to parameters and read out variable values after solving, or wrap the whole
    thing in a CvxpyLayer via build_layer().
    """
    # Variables -- what the solver chooses. `shortfall` is an auxiliary variable
    # standing in for max(0, demand_mean - (q+s)) -- see module docstring for why
    # cp.pos() can't be used directly here without breaking DPP.
    q = cp.Variable(nonneg=True, name="order_quantity")
    s = cp.Variable(nonneg=True, name="safety_stock")
    shortfall = cp.Variable(nonneg=True, name="shortfall")

    # Parameters -- never hardcode these, or the layer stops being differentiable
    demand_mean = cp.Parameter(nonneg=True, name="demand_mean")
    unit_holding_cost = cp.Parameter(nonneg=True, name="unit_holding_cost")
    unit_stockout_penalty = cp.Parameter(nonneg=True, name="unit_stockout_penalty")
    unit_cost = cp.Parameter(nonneg=True, name="unit_cost")
    budget = cp.Parameter(nonneg=True, name="budget")
    capacity = cp.Parameter(nonneg=True, name="capacity")

    # Objective: holding cost on safety stock + ordering cost on order qty
    # + expected stockout penalty on the shortfall, PLUS a tiny quadratic
    # regularization term (REG_EPS). Without it, this is a pure LP, and pure
    # LPs generically land at a VERTEX of the feasible region (e.g. q exactly
    # 0) -- verified empirically, gradcheck fails there because the solution
    # can jump discontinuously between vertices under an infinitesimal
    # parameter nudge. Adding a strongly-convex quadratic term makes the
    # problem a QP with a single, smooth, uniquely-differentiable optimum,
    # while changing the actual optimal q/s by a negligible amount (REG_EPS
    # is small). REG_EPS is a plain Python float, not a cp.Parameter, so this
    # does not introduce any new DPP risk.
    objective = cp.Minimize(
        unit_holding_cost * s
        + unit_cost * q
        + unit_stockout_penalty * shortfall
        + REG_EPS * cp.sum_squares(q)
        + REG_EPS * cp.sum_squares(s)
        + REG_EPS * cp.sum_squares(shortfall)
    )

    # Constraints
    constraints = [
        q + s <= capacity,
        q * unit_cost <= budget,
        shortfall >= demand_mean - (q + s),
    ]

    problem = cp.Problem(objective, constraints)

    parameters = {
        "demand_mean": demand_mean,
        "unit_holding_cost": unit_holding_cost,
        "unit_stockout_penalty": unit_stockout_penalty,
        "unit_cost": unit_cost,
        "budget": budget,
        "capacity": capacity,
    }
    variables = {"order_quantity": q, "safety_stock": s, "shortfall": shortfall}
    return problem, parameters, variables


def build_layer():
    """
    Wraps build_problem() in a cvxpylayers CvxpyLayer, making it usable as a
    differentiable PyTorch layer: layer(*param_tensors) -> (q_tensor, s_tensor),
    with gradients of q, s w.r.t. every parameter tensor available via ordinary
    autograd.

    Asserts DPP compliance before wrapping -- this is required, not optional:
    a non-DPP problem wraps into a CvxpyLayer without raising an error, but
    produces zero (wrong) gradients silently.
    """
    from cvxpylayers.torch import CvxpyLayer

    problem, parameters, variables = build_problem()
    assert problem.is_dcp(dpp=True), (
        "optimization/core.py: problem is not DPP-compliant -- differentiable "
        "gradients would be silently wrong. Check for parameter*parameter products."
    )
    layer = CvxpyLayer(
        problem,
        parameters=list(parameters.values()),
        variables=list(variables.values()),
    )
    return layer, parameters, variables


def compute_reorder_point(demand_mean, demand_std, lead_time_days, z_score: float = Z_SCORE_95):
    """
    r = mu*L + z*sigma*sqrt(L)

    Deliberately kept OUTSIDE the CVXPY problem (see module docstring) -- this is
    plain arithmetic, not an optimization. Works transparently with either plain
    floats (fallback path) or torch tensors with requires_grad=True (differentiable
    path): ordinary autograd already handles gradients through +, *, and ** here,
    no CvxpyLayer required for this piece.
    """
    lead_time_sqrt = lead_time_days ** 0.5
    return demand_mean * lead_time_days + z_score * demand_std * lead_time_sqrt
