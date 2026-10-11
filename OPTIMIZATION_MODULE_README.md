# Optimization Engine -- `feature/optimization`

Member 2 (Bhuvaneshwari M, 1VJ23CS009) build: a differentiable inventory
optimizer using CVXPY + cvxpylayers, exposed as
`optimize_inventory(forecast, constraints)`.

## Setup

```
pip install -r requirements.txt --break-system-packages   # or use a venv
```

## What it does

Given a demand forecast (expected demand, uncertainty bounds, model
confidence) and a set of business constraints (budget, warehouse capacity,
lead time, unit costs), this module computes three numbers:

- **order_quantity** -- how many units to order right now
- **safety_stock** -- extra buffer stock held against demand uncertainty
- **reorder_point** -- the inventory level that should trigger a new order

Unlike a standard optimizer that only returns one answer, this one is
**differentiable**: the optimal `order_quantity` and `safety_stock` can be
differentiated with respect to every input parameter (demand, costs, budget,
capacity), so the optimization can sit inside a larger trainable pipeline
rather than being a disconnected black-box step.

## Files

```
optimization/
  core.py       # builds the CVXPY problem, wraps it as a differentiable
                 # PyTorch layer via cvxpylayers
  fallback.py   # plain non-differentiable solve, used if the layer fails
  main.py       # entry point: optimize_inventory(forecast, constraints)
```

## Real bugs found and fixed

1. **Objective bug** -- the ordering-cost term used `unit_holding_cost`
   twice instead of `unit_cost`.
2. **Two DPP violations** -- CVXPY requires parameters to combine only
   affinely with each other to stay differentiable. Two disguised
   parameter*parameter products were found (in the original reorder-point
   constraint, and hidden inside a `cp.pos()` call) and fixed. Verified with
   `problem.is_dcp(dpp=True)` -- this class of bug compiles and solves fine,
   but silently returns wrong gradients.
3. **LP vertex instability** -- even after fixing DPP compliance,
   `torch.autograd.gradcheck` still failed, because the un-regularized
   problem is a pure LP and lands at a vertex of the feasible region (a
   non-smooth point). Fixed with a small quadratic regularization term,
   turning the LP into a QP with a unique, smooth, differentiable optimum.

Full derivation and DPP notes: `docs/optimization_formulation.md`

## Proof of differentiability

```
python -m pytest tests/ -v
```

`tests/test_gradients.py` verifies via `torch.autograd.gradcheck` (analytical
gradient checked against finite differences) that the optimizer's outputs
are genuinely differentiable w.r.t. every input parameter. All 5 tests pass.

## Verified end-to-end on real data

```
python smoke_test.py     # sanity check: imports + DPP compliance
python run_pipeline.py   # real forecast_demand() -> real optimize_inventory()
```

Ran against the real `retail_store_inventory.csv` (73,100 rows, 731 days,
100 SKUs): a real forecast from the forecasting module feeds directly into
`optimize_inventory()`, returning `solver_status="optimal"` -- confirming
the two modules' contracts genuinely integrate end-to-end.

---

## Ownership

Built as part of a 4-member VTU group project (AI-powered supply chain
optimization system), guided by Prof. Supriya, Dept. of CSE, VVIT. This repo
isolates my individual contribution for submission and viva prep.
`forecasting/` and `shared/` are included unmodified, only so the pipeline
runs end-to-end.

| Path | Owner |
|---|---|
| `optimization/` | **Bhuvaneshwari M (1VJ23CS009)** |
| `tests/test_gradients.py` | **Bhuvaneshwari M (1VJ23CS009)** |
| `docs/optimization_formulation.md` | **Bhuvaneshwari M (1VJ23CS009)** |
| `forecasting/` | D Sam Goldwin (1VJ23CS013) |
| `shared/schemas.py` | Team (shared contract) |
| `data/retail_store_inventory.csv` | Team (shared dataset) |
