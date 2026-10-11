"""Backtest: train on the first 70% of days, tune on the next 15%, report on the final 15% (never seen)."""
import time

import numpy as np

from . import features, forecast
from .policies import BaseStockPolicy, CvxpyPolicy, RulePolicy
from .rl import ACTIONS, RLPolicy
from .simulator import SimParams, simulate

TAU_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


def _f(x):
    return float(np.round(x, 4))


def tune_tau(F, model, panel, a, b, params, init):
    """Pick the service level (quantile) that gives the lowest simulated cost on VALIDATION days."""
    T, N = panel.T, panel.N
    Xall = F["X"].reshape(-1, F["X"].shape[2])
    best = (None, np.inf)
    table = []
    for tau in TAU_GRID:
        S = model.quantile(Xall, tau).reshape(T, N) * F["H"] * F["scale"]
        c = simulate(panel.demand, a, b, BaseStockPolicy(S), params, init).total_cost
        table.append((tau, c))
        if c < best[1]:
            best = (tau, c)
    return best[0], table


def run_dataset(panel, holding=1.0, stockout=5.0, lead=3, tight_share=0.85, n_scen=40,
                rl_episodes=6000, seed=0, log=print):
    t_start = time.time()
    a, b = panel.split()
    T, N, H = panel.T, panel.N, lead + 1
    F = features.build(panel, H, a)
    scale = F["scale"]
    Xtr, ytr, _ = features.rows(F, 0, a)
    Xv, yv, _ = features.rows(F, a, b)
    Xte, yte, _ = features.rows(F, b, T)
    Xall = F["X"].reshape(-1, F["X"].shape[2])
    w = panel.unit_value
    base = SimParams(holding, stockout, lead, weights=w)
    init = scale * H

    log(f"[{panel.name}] fitting forecaster on {a} training days")
    mse = forecast.MSEForecaster().fit(Xtr, ytr).calibrate(Xv, yv)
    tau, tau_table = tune_tau(F, mse, panel, a, b, base, init)
    # planning overage cost that makes the newsvendor ratio equal the tuned service level
    h_eff = stockout * (1 - tau) / tau
    log(f"[{panel.name}] tuned service level on validation days: {tau:.2f}")

    dfl = forecast.DecisionFocused(h_eff, stockout).fit(Xtr, ytr, Xv, yv, mse)
    S_mse = mse.quantile(Xall, tau).reshape(T, N) * H * scale
    S_dfl = dfl.predict(Xall).reshape(T, N) * H * scale
    yhat = mse.predict(Xall).reshape(T, N)

    # ---- forecast quality on the untouched test days
    naive = np.nan_to_num(Xte[:, 4], nan=1.0)              # mean of the last 28 days
    lo, hi = mse.interval(Xte, 0.8)
    quality = [
        {"model": "Last-28-day average (naive)", "wape_pct": _f(100 * forecast.wape(naive, yte)), "coverage_pct": None,
         "newsvendor_cost": None},
        {"model": "Ridge, squared-error loss + conformal range", "wape_pct": _f(100 * forecast.wape(mse.predict(Xte), yte)),
         "coverage_pct": _f(100 * np.mean((yte >= lo) & (yte <= hi))),
         "newsvendor_cost": _f(forecast.newsvendor_cost(mse.quantile(Xte, tau), yte, h_eff, stockout))},
        {"model": "Ridge, decision-focused loss", "wape_pct": None, "coverage_pct": None,
         "newsvendor_cost": _f(forecast.newsvendor_cost(dfl.predict(Xte), yte, h_eff, stockout))},
    ]

    # ---- capacity scenarios
    mean_total_target = float(np.mean(S_mse[b:T].sum(axis=1)))
    scenarios = {"ample": None, "tight": tight_share * mean_total_target}
    results = {}
    curves = {}
    shadow = {}
    rl_share = {}
    rl = RLPolicy(S_mse, panel.demand, H, holding, stockout, lead, seed=seed)
    log(f"[{panel.name}] training RL agent ({rl_episodes} episodes)")
    rl.train(features.MIN_HISTORY + 8, a, episodes=rl_episodes)

    for sc_name, cap in scenarios.items():
        p = SimParams(holding, stockout, lead, capacity=cap, weights=w)
        cv = CvxpyPolicy(yhat, scale, H, mse.resid, h_eff, stockout, capacity=cap, n_scen=n_scen, weights=w)
        pols = {
            "Simple rule": RulePolicy(scale, lead),
            "Forecast base-stock (squared-error)": BaseStockPolicy(S_mse),
            "Forecast base-stock (decision-focused)": BaseStockPolicy(S_dfl),
            "CVXPY optimizer": cv,
            "RL correction": rl,
        }
        results[sc_name] = {}
        curves[sc_name] = {}
        for name, pol in pols.items():
            log(f"[{panel.name}] simulating {sc_name} capacity: {name}")
            if pol is rl:
                rl.action_counts[:] = 0
            r = simulate(panel.demand, b, T, pol, p, init)
            results[sc_name][name] = r.summary()
            if pol is rl:
                rl_share[sc_name] = [_f(100 * c / max(rl.action_counts.sum(), 1)) for c in rl.action_counts]
            curves[sc_name][name] = [_f(v) for v in r.stock_series.sum(axis=1)]
        if cv.shadow:
            sh = np.array(cv.shadow)
            shadow[sc_name] = {"mean": _f(sh.mean()), "max": _f(sh.max()), "days_binding_pct": _f(100 * np.mean(sh > 1e-6)),
                               "capacity": _f(cap)}
        else:
            shadow[sc_name] = {"mean": 0.0, "max": 0.0, "days_binding_pct": 0.0, "capacity": None}
        results[sc_name]["_solver_ok_pct"] = _f(100 * np.mean([s == "optimal" for s in cv.statuses]))

    def savings(sc):
        rule = results[sc]["Simple rule"]["total_cost"]
        best_name = min((n for n in results[sc] if not n.startswith("_")), key=lambda n: results[sc][n]["total_cost"])
        best = results[sc][best_name]["total_cost"]
        return {"best_policy": best_name, "best_cost": best, "rule_cost": rule, "savings_pct": _f(100 * (rule - best) / rule)}

    return {
        "name": panel.name, "n_skus": N, "days": T,
        "split": {"train_days": a, "validation_days": b - a, "test_days": T - b},
        "settings": {"holding": holding, "stockout": stockout, "lead_days": lead, "tuned_service_level": tau,
                     "effective_holding_cost": _f(h_eff), "tight_capacity_share": tight_share, "rl_episodes": rl_episodes},
        "tau_tuning": [{"service_level": t, "validation_cost": _f(c)} for t, c in tau_table],
        "forecast_quality": quality,
        "policies": results,
        "savings": {sc: savings(sc) for sc in scenarios},
        "shadow_prices": shadow,
        "rl": {"curve": rl.curve, "episodes": rl_episodes, "actions": [float(a) for a in ACTIONS],
               "action_share_pct": rl_share, "states": 24},
        "stock_curves": curves,
        "runtime_s": _f(time.time() - t_start),
    }
