"""Ordering policies. Every policy exposes order(t, on_hand, position) -> order quantities per SKU."""
import cvxpy as cp
import numpy as np


class RulePolicy:
    """The classic shop rule: when position falls to the average lead-time demand, refill to a few extra days."""

    def __init__(self, mean_daily, lead, extra_days=3):
        self.s = mean_daily * (lead + 1)
        self.S = mean_daily * (lead + 1 + extra_days)

    def order(self, t, on_hand, pos):
        return np.where(pos <= self.s, self.S - pos, 0.0)


class BaseStockPolicy:
    """Order up to a target S[t] each day. The target comes from a forecaster (MSE+conformal or decision-focused)."""

    def __init__(self, target):
        self.target = target          # (T, N) in units

    def order(self, t, on_hand, pos):
        return np.maximum(self.target[t] - pos, 0.0)


class CvxpyPolicy:
    """Scenario-based newsvendor LP over all SKUs with a SHARED capacity (sample average approximation).

    min  mean_s sum_i [ h*(pos_i+q_i-D_is)+ + p*(D_is-pos_i-q_i)+ ]   s.t.  q >= 0,  sum_i (pos_i+q_i) <= capacity
    D_is = (forecast_i + calibrated residual_s) over the protection interval. The dual value of the capacity
    constraint is the shadow price: the cost saved by one more unit of storage.
    """

    def __init__(self, yhat_norm, scale, H, resid_norm, holding, stockout, capacity=None, n_scen=40, weights=None):
        self.yhat, self.scale, self.H = yhat_norm, scale, H
        qs = (np.arange(n_scen) + 0.5) / n_scen
        self.resid = np.quantile(resid_norm, qs)                   # scenario errors from the calibration residuals
        n, ns = len(scale), n_scen
        self.n, self.ns = n, ns
        self.pos = cp.Parameter(n, nonneg=True)
        self.D = cp.Parameter((ns, n), nonneg=True)
        self.cap = cp.Parameter(nonneg=True) if capacity is not None else None
        self.q = cp.Variable(n, nonneg=True)
        total = cp.reshape(self.pos + self.q, (1, n), order="C")
        stock = np.ones((ns, 1)) @ total
        w = np.ones(n) if weights is None else np.asarray(weights, float)
        hw = np.ones((ns, 1)) @ (holding * w)[None, :]
        pw = np.ones((ns, 1)) @ (stockout * w)[None, :]
        cost = cp.multiply(hw, cp.pos(stock - self.D)) + cp.multiply(pw, cp.pos(self.D - stock))
        obj = cp.Minimize(cp.sum(cost) / ns)
        cons = [cp.sum(self.pos + self.q) <= self.cap] if capacity is not None else []
        self.capacity = capacity
        self.prob = cp.Problem(obj, cons)
        self.cap_con = cons[0] if cons else None
        self.shadow = []        # one value per day
        self.statuses = []

    def order(self, t, on_hand, pos):
        d = (self.yhat[t][None, :] + self.resid[:, None]) * (self.H * self.scale)[None, :]
        self.D.value = np.maximum(d, 0.0)
        self.pos.value = np.maximum(pos, 0.0)
        if self.cap is not None:
            self.cap.value = float(self.capacity)
        try:
            self.prob.solve(solver=cp.CLARABEL)
        except cp.SolverError:
            self.prob.solve()
        self.statuses.append(self.prob.status)
        if self.q.value is None:
            return np.zeros(self.n)
        if self.cap_con is not None and self.cap_con.dual_value is not None:
            self.shadow.append(float(np.asarray(self.cap_con.dual_value).ravel()[0]))
        return np.maximum(self.q.value, 0.0)
