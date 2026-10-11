# Optimization Formulation
Owner: Bhuvaneshwari M (1VJ23CS009)

## Variables
- q: order quantity (cp.Variable, nonneg=True)
- s: safety stock (cp.Variable, nonneg=True)
- shortfall: auxiliary variable standing in for max(0, demand_mean - (q+s)) (cp.Variable, nonneg=True)
- r: reorder point -- NOT a solver variable, computed separately (see below)

## Parameters
- demand_mean: mean expected demand over the forecast horizon (from forecasting.forecast_demand)
- demand_std: demand standard deviation, derived from the forecast's upper/lower bound spread
- unit_holding_cost, unit_stockout_penalty, unit_cost: business cost inputs
- budget, capacity: business constraint inputs
- lead_time_days: supplier lead time

## Objective
```
minimize  unit_holding_cost * s
        + unit_cost * q
        + unit_stockout_penalty * shortfall
        + REG_EPS * (q^2 + s^2 + shortfall^2)
```

The REG_EPS quadratic term (1e-4) is a strong-convexity regularizer, added after
discovering empirically that the un-regularized pure LP lands at a vertex of the
feasible region (e.g. q = 0 exactly), which causes torch.autograd.gradcheck to fail:
LP solutions are generically non-smooth at vertices, so an infinitesimal parameter
perturbation can jump the solution discontinuously. Adding REG_EPS turns the LP into
a QP with a unique, smooth optimum, fixing gradcheck, while changing the optimal
q/s by a negligible amount.

## Constraints
```
q + s <= capacity
q * unit_cost <= budget
shortfall >= demand_mean - (q + s)
```

`shortfall` replaces a direct `cp.pos(demand_mean - (q+s))` term in the objective.
Both formulations are mathematically equivalent at the optimum (minimizing a
positive multiple of shortfall pushes it down to exactly max(0, demand_mean-(q+s))),
but only the auxiliary-variable form is DPP-compliant -- see DPP notes below.

## Reorder point (computed outside CVXPY)
```
r = demand_mean * lead_time_days + z_score * demand_std * sqrt(lead_time_days)
```
z_score = 1.65 for a 95% service level.

r is deliberately NOT a CVXPY variable/constraint. It is pure parameter arithmetic
-- the solver has no actual choice to make about r -- so it is computed directly
in optimization/core.py::compute_reorder_point(), which works transparently on
both plain floats (fallback path) and torch tensors with requires_grad=True
(differentiable path), since ordinary autograd already handles +, *, ** correctly.

## DPP compliance notes (verified empirically, see tests/test_gradients.py::test_dpp_compliance)

Two real DPP violations were found and fixed during development:

1. **r as a CVXPY equality constraint** (original formulation):
   `r == demand_mean * lead_time_days + z*demand_std*sqrt(lead_time_days)`
   Both `demand_mean * lead_time_days` (parameter * parameter) and
   `demand_std * sqrt(lead_time_days)` (parameter * nonlinear-function-of-parameter)
   are disguised parameter*parameter products. DPP requires parameters to combine
   only affinely with each other; this fails `problem.is_dcp(dpp=True)` even though
   `problem.is_dcp()` (ordinary convexity, ignoring DPP) returns True -- i.e. the
   problem solves fine but a CvxpyLayer wrapping it would silently return wrong
   gradients. **Fix:** removed from the CVXPY problem entirely (see above).

2. **cp.pos(demand_mean - (q+s)) in the objective, multiplied by unit_stockout_penalty**:
   This traps a parameter (demand_mean) inside a nonlinear atom (pos), and
   multiplying the result by another parameter (unit_stockout_penalty) is again a
   disguised parameter*parameter product, one level deeper than issue 1.
   **Fix:** standard LP reformulation -- replaced with an auxiliary `shortfall`
   variable and an inequality constraint (see Constraints above). This is
   mathematically equivalent at the optimum and is DPP-compliant since
   `unit_stockout_penalty * shortfall` is (parameter * variable), and
   `shortfall >= demand_mean - (q+s)` has demand_mean appearing affinely, alone.

`problem.is_dcp(dpp=True)` returns `True` after both fixes.

## Service level / z-score justification
z = 1.65 corresponds to a 95% cycle service level under the standard normal-
approximation safety stock formula (P(demand during lead time <= r) = 0.95).
This is a common default in inventory theory; swapping to a different target
service level only requires changing Z_SCORE_95 in optimization/core.py.
