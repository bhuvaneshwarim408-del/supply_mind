"""Daily inventory simulator shared by backtests, RL training and the what-if page.

Each day t:  (1) orders due today arrive  (2) the policy orders (arrives `lead` days later)
             (3) demand happens; unmet demand is lost  (4) cost = holding*stock + stockout*lost units.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimParams:
    holding: float = 1.0
    stockout: float = 5.0
    lead: int = 3                 # lead time the policies PLAN with
    actual_lead: int = None       # lead time that really happens (supplier delay); defaults to `lead`
    capacity: float = None        # cap on total inventory position (stock + on order) across SKUs
    weights: np.ndarray = None    # (N,) relative unit value of each SKU; scales both costs

    @property
    def H(self):
        return self.lead + 1


@dataclass
class SimResult:
    holding_cost: float
    stockout_cost: float
    fill_rate: float
    avg_stock: float
    orders: float
    stock_series: np.ndarray = field(repr=False)      # (days, N)
    total_cost: float = 0.0

    def summary(self):
        return {
            "total_cost": round(self.total_cost, 1), "holding_cost": round(self.holding_cost, 1),
            "stockout_cost": round(self.stockout_cost, 1), "fill_rate_pct": round(100 * self.fill_rate, 2),
            "avg_stock": round(self.avg_stock, 1),
        }


def apply_capacity(q, pos, capacity):
    """Naive allocation used by every policy that is not capacity-aware: scale all orders down together."""
    if capacity is None:
        return q
    room = max(0.0, capacity - pos.sum())
    total = q.sum()
    return q * (room / total) if total > room and total > 0 else q


def simulate(demand, t0, t1, policy, p: SimParams, init_stock):
    T, N = demand.shape
    lead = p.actual_lead if p.actual_lead is not None else p.lead
    arrivals = np.zeros((T + lead + 2, N))
    w = p.weights if p.weights is not None else np.ones(N)
    on_hand = init_stock.astype(float).copy()
    hold = short = lost_units = demand_units = orders = 0.0
    series = np.zeros((t1 - t0, N))
    for t in range(t0, t1):
        on_hand += arrivals[t]
        pos = on_hand + arrivals[t + 1:t + lead + 1].sum(axis=0)       # stock plus everything still on order
        q = np.maximum(policy.order(t, on_hand, pos), 0.0)
        q = apply_capacity(q, pos, p.capacity)
        arrivals[t + lead] += q
        orders += q.sum()
        sales = np.minimum(on_hand, demand[t])
        lost = demand[t] - sales
        on_hand -= sales
        hold += p.holding * float(w @ on_hand)
        short += p.stockout * float(w @ lost)
        lost_units += lost.sum()
        demand_units += demand[t].sum()
        series[t - t0] = on_hand
    return SimResult(hold, short, 1 - lost_units / max(demand_units, 1e-9), float(series.mean()), orders, series, hold + short)
