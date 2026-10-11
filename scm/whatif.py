"""What-if runs for one SKU: demand spike, supplier delay, storage limit. Uses the same simulator as the backtests."""
import os
import threading

import numpy as np

from . import data, features, forecast
from .backtest import tune_tau
from .policies import BaseStockPolicy, CvxpyPolicy, RulePolicy
from .simulator import SimParams, simulate

LEAD = 3
HOLDING, STOCKOUT = 1.0, 5.0
WINDOW = 60
_lock = threading.Lock()
_art = None


def _panel():
    csv = os.environ.get("SUPPLYMIND_CSV", "data/retail_store_inventory.csv")
    if os.path.exists(csv):
        return data.load_panel(csv, name="Retail CSV")
    return data.synthetic_panel(n_skus=40, name="Synthetic (CSV not found)")


def artifacts():
    global _art
    with _lock:
        if _art is None:
            P = _panel()
            a, b = P.split()
            H = LEAD + 1
            F = features.build(P, H, a)
            Xtr, ytr, _ = features.rows(F, 0, a)
            Xv, yv, _ = features.rows(F, a, b)
            mse = forecast.MSEForecaster().fit(Xtr, ytr).calibrate(Xv, yv)
            base = SimParams(HOLDING, STOCKOUT, LEAD, weights=P.unit_value)
            tau, _ = tune_tau(F, mse, P, a, b, base, F["scale"] * H)
            Xall = F["X"].reshape(-1, F["X"].shape[2])
            _art = {"panel": P, "F": F, "mse": mse, "tau": tau, "H": H, "h_eff": STOCKOUT * (1 - tau) / tau,
                    "S": mse.quantile(Xall, tau).reshape(P.T, P.N) * H * F["scale"],
                    "yhat": mse.predict(Xall).reshape(P.T, P.N)}
        return _art


def sku_list():
    return list(artifacts()["panel"].skus)


def run(sku, demand_change_pct=0.0, delay_days=0, capacity=None):
    art = artifacts()
    P, F, H = art["panel"], art["F"], art["H"]
    i = P.skus.index(sku)
    T = P.T
    t0 = T - WINDOW
    demand = P.demand[:, [i]].copy()
    demand[t0:] *= 1 + demand_change_pct / 100.0                    # surprise the forecast does not know about
    scale = F["scale"][[i]]
    params = SimParams(HOLDING, STOCKOUT, LEAD, actual_lead=LEAD + int(delay_days), capacity=capacity,
                       weights=P.unit_value[[i]])
    init = scale * H
    cv = CvxpyPolicy(art["yhat"][:, [i]], scale, H, art["mse"].resid, art["h_eff"], STOCKOUT, capacity=capacity,
                     n_scen=40, weights=P.unit_value[[i]])
    pols = {
        "Simple rule": RulePolicy(scale, LEAD),
        "Forecast base-stock": BaseStockPolicy(art["S"][:, [i]]),
        "CVXPY optimizer": cv,
    }
    out = {"sku": sku, "dates": [str(d)[:10] for d in P.dates[t0:]], "policies": {}}
    for name, pol in pols.items():
        r = simulate(demand, t0, T, pol, params, init)
        out["policies"][name] = {"stock": [round(float(v), 1) for v in r.stock_series[:, 0]], **r.summary()}
    out["settings"] = {"demand_change_pct": demand_change_pct, "delay_days": int(delay_days), "capacity": capacity,
                       "planned_lead_days": LEAD, "window_days": WINDOW, "service_level": art["tau"]}
    return out
