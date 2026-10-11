"""
tests/test_gradients.py
Owner: Bhuvaneshwari M (1VJ23CS009)

This is the evidence for the project title's "differentiable" claim. If this test
doesn't pass (or doesn't exist), the differentiability of the optimization layer
is asserted, not verified -- exactly the DPP-violation trap the build spec warns
about in optimization/core.py.

torch.autograd.gradcheck works by comparing the layer's analytically-computed
gradient (via CvxpyLayer's implicit differentiation through the solver) against
a numerical gradient obtained by finite differences (nudging each input by a
tiny eps and observing how much the output moves). If they agree to within a
tight tolerance, the analytical gradient is verified correct -- not just
"present", but numerically trustworthy.
"""

import pytest
import torch

from optimization.core import build_layer, build_problem


def test_dpp_compliance():
    """
    The single most important sanity check in this whole module: a DPP violation
    compiles and even solves fine, but CvxpyLayer will silently produce wrong
    (often zero) gradients. This must be True before any gradient claim is
    trusted.
    """
    problem, params, variables = build_problem()
    assert problem.is_dcp(dpp=True) is True


def test_layer_forward_matches_plain_solve():
    """Sanity check: the differentiable layer's forward pass should agree with
    a plain (non-differentiable) solve of the same problem for the same inputs."""
    problem, params, variables = build_problem()
    params["demand_mean"].value = 100.0
    params["unit_holding_cost"].value = 2.0
    params["unit_stockout_penalty"].value = 10.0
    params["unit_cost"].value = 5.0
    params["budget"].value = 5000.0
    params["capacity"].value = 500.0
    problem.solve()
    plain_q = variables["order_quantity"].value
    plain_s = variables["safety_stock"].value

    layer, layer_params, layer_vars = build_layer()
    inputs = [
        torch.tensor(100.0, dtype=torch.float64),  # demand_mean
        torch.tensor(2.0, dtype=torch.float64),    # unit_holding_cost
        torch.tensor(10.0, dtype=torch.float64),   # unit_stockout_penalty
        torch.tensor(5.0, dtype=torch.float64),    # unit_cost
        torch.tensor(5000.0, dtype=torch.float64), # budget
        torch.tensor(500.0, dtype=torch.float64),  # capacity
    ]
    q_t, s_t, shortfall_t = layer(*inputs)

    assert q_t.item() == pytest.approx(plain_q, abs=1e-3)
    assert s_t.item() == pytest.approx(plain_s, abs=1e-3)


def test_gradcheck_demand_mean():
    """
    The core differentiability proof: gradient of order_quantity w.r.t. demand_mean,
    verified via finite differences. Other parameters are held fixed (not the
    variable under test) so gradcheck perturbs demand_mean alone.
    """
    layer, params, variables = build_layer()

    demand_mean = torch.tensor(100.0, dtype=torch.float64, requires_grad=True)
    fixed = [
        torch.tensor(2.0, dtype=torch.float64),    # unit_holding_cost
        torch.tensor(10.0, dtype=torch.float64),   # unit_stockout_penalty
        torch.tensor(5.0, dtype=torch.float64),    # unit_cost
        torch.tensor(5000.0, dtype=torch.float64), # budget
        torch.tensor(500.0, dtype=torch.float64),  # capacity
    ]

    def f(dm):
        q, s, shortfall = layer(dm, *fixed)
        return q

    assert torch.autograd.gradcheck(f, (demand_mean,), eps=1e-6, atol=1e-4)


def test_gradcheck_unit_holding_cost():
    """Same proof, this time perturbing unit_holding_cost -- confirms the gradient
    check isn't accidentally passing for just one parameter."""
    layer, params, variables = build_layer()

    unit_holding_cost = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    demand_mean = torch.tensor(100.0, dtype=torch.float64)
    fixed_tail = [
        torch.tensor(10.0, dtype=torch.float64),   # unit_stockout_penalty
        torch.tensor(5.0, dtype=torch.float64),    # unit_cost
        torch.tensor(5000.0, dtype=torch.float64), # budget
        torch.tensor(500.0, dtype=torch.float64),  # capacity
    ]

    def f(uhc):
        q, s, shortfall = layer(demand_mean, uhc, *fixed_tail)
        return s

    assert torch.autograd.gradcheck(f, (unit_holding_cost,), eps=1e-6, atol=1e-4)


def test_reorder_point_gradient_flows():
    """
    compute_reorder_point() is plain arithmetic (deliberately outside CVXPY, see
    optimization/core.py docstring) -- confirm ordinary torch autograd still
    produces a non-None, finite gradient through it, since it feeds the same
    OptimizationOutput.reorder_point field.
    """
    from optimization.core import compute_reorder_point

    demand_mean = torch.tensor(100.0, dtype=torch.float64, requires_grad=True)
    demand_std = torch.tensor(15.0, dtype=torch.float64, requires_grad=True)
    lead_time_days = torch.tensor(7.0, dtype=torch.float64, requires_grad=True)

    r = compute_reorder_point(demand_mean, demand_std, lead_time_days)
    r.backward()

    assert demand_mean.grad is not None and torch.isfinite(demand_mean.grad)
    assert demand_std.grad is not None and torch.isfinite(demand_std.grad)
    assert lead_time_days.grad is not None and torch.isfinite(lead_time_days.grad)
