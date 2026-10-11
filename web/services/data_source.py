"""Data layer for the UI.

If the team's modules (forecasting, optimization, shared) and the CSV are found,
every page shows REAL results from them (see real_backend.py). Otherwise it
falls back to deterministic DEMO data so the UI still runs. Set the
environment variable SUPPLYMIND_DEMO=1 to force demo data.
"""
import math
import random
from datetime import date, timedelta

from . import real_backend

IS_DEMO = not real_backend.available()
DEFAULT_HORIZON = real_backend.DEFAULT_HORIZON
DEFAULT_CONSTRAINTS = real_backend.DEFAULT_CONSTRAINTS
SKUS = [f"P{i:04d}" for i in range(1, 21)] if IS_DEMO else real_backend.skus()
Z = 1.65  # about a 95% service level (demo only)


# ---------------- public API ----------------
def forecast(sku, horizon=DEFAULT_HORIZON):
    return _demo_forecast(sku, horizon=horizon) if IS_DEMO else real_backend.forecast_view(sku, horizon)


def optimizer(sku, horizon=DEFAULT_HORIZON, constraints=None, with_explanation=True, generate=True):
    if IS_DEMO:
        return _demo_optimizer(sku)
    return real_backend.optimizer_view(sku, horizon, constraints or DEFAULT_CONSTRAINTS, with_explanation, generate)


def portfolio():
    return _demo_portfolio() if IS_DEMO else real_backend.portfolio_view()


# ---------------- demo data ----------------
def _profile(sku):
    r = random.Random(sku)
    return {
        "base": r.randint(30, 140),
        "amp": r.uniform(0.12, 0.35),
        "noise": r.uniform(0.06, 0.18),
        "lead": r.randint(2, 7),
        "hold": r.uniform(0.4, 2.0) / 30,
        "order": r.uniform(30, 90),
        "stockout": r.uniform(3, 9),
    }


def _level(p, i):
    return p["base"] * (1 + p["amp"] * math.sin(2 * math.pi * i / 7))


def _demo_forecast(sku, history=45, horizon=14):
    p = _profile(sku)
    r = random.Random(sku + ":f")
    sigma = p["base"] * p["noise"]
    start = date.today() - timedelta(days=history)
    dates, actual, pred, lower, upper = [], [], [], [], []
    for i in range(history + horizon):
        lvl = _level(p, i)
        dates.append((start + timedelta(days=i)).isoformat())
        actual.append(max(0, round(lvl + r.gauss(0, sigma))) if i < history else None)
        pred.append(round(lvl))
        width = 1.28 * sigma * math.sqrt(1 + max(0, i - history + 1) / 7)
        lower.append(max(0, round(lvl - width)))
        upper.append(round(lvl + width))
    recent = range(history - 14, history)
    wape = sum(abs(actual[i] - pred[i]) for i in recent) / max(1, sum(actual[i] for i in recent))
    fut = range(history, history + horizon)
    mean = sum(pred[i] for i in fut) / horizon
    half = sum((upper[i] - lower[i]) / 2 for i in fut) / horizon
    return {
        "sku": sku, "dates": dates, "actual": actual, "forecast": pred,
        "lower": lower, "upper": upper, "split_index": history,
        "metrics": {
            "range_pct": round(100 * half / mean, 1) if mean else 0.0,
            "certainty_pct": round(max(0.0, 100 - 100 * half / mean), 1) if mean else 0.0,
            "lower_avg": round(sum(lower[i] for i in fut) / horizon), "upper_avg": round(sum(upper[i] for i in fut) / horizon),
            "confidence_pct": round(max(0.0, 100 - wape * 100), 1),
            "horizon_total": sum(pred[i] for i in fut),
            "avg_daily": round(mean),
            "horizon_days": horizon,
        },
    }


def _demo_optimizer(sku, days=60):
    p = _profile(sku)
    r = random.Random(sku + ":o")
    mu, sigma, lead = p["base"], p["base"] * p["noise"], p["lead"]
    safety = round(Z * sigma * math.sqrt(lead))
    rop = round(mu * lead + safety)
    qty = round(math.sqrt(2 * mu * p["order"] / p["hold"]))

    stock, pending, series = rop + qty // 2, [], []
    hold = ordering = short_cost = 0.0
    short_units = demand_total = orders = 0
    for i in range(days):
        stock += sum(q for a, q in pending if a == i)
        pending = [(a, q) for a, q in pending if a != i]
        demand = max(0, round(_level(p, i) + r.gauss(0, sigma)))
        sold = min(stock, demand)
        stock -= sold
        demand_total += demand
        short_units += demand - sold
        short_cost += (demand - sold) * p["stockout"]
        hold += stock * p["hold"]
        if stock <= rop and not pending:
            pending.append((i + lead, qty))
            orders += 1
            ordering += p["order"]
        series.append(stock)

    start = date.today() - timedelta(days=days)
    if series[-1] <= rop:
        status = "On order" if pending else "Reorder now"
    else:
        status = "Low" if series[-1] <= rop * 1.3 else "Healthy"
    return {
        "sku": sku, "projected": False,
        "dates": [(start + timedelta(days=i)).isoformat() for i in range(days)],
        "stock": series, "reorder_point": rop, "safety_stock": safety, "order_qty": qty,
        "lead_time_days": lead, "orders_placed": orders,
        "projected_service_pct": round(100 * (1 - short_units / max(1, demand_total)), 1),
        "total_cost": round(hold + ordering + short_cost),
        "solver_status": "demo",
        "confidence_pct": _demo_forecast(sku)["metrics"]["confidence_pct"],
        "certainty_pct": _demo_forecast(sku)["metrics"]["certainty_pct"],
        "avg_daily": mu, "horizon_days": days,
        "stock_now": series[-1], "days_of_cover": round(series[-1] / mu, 1),
        "status": status, "explanation": None,
    }


def _demo_portfolio():
    rows = []
    for sku in SKUS:
        r = _demo_optimizer(sku)
        rows.append({"sku": sku, "cost": r["total_cost"], "qty": r["order_qty"], "rop": r["reorder_point"],
                     "stock_now": r["stock_now"], "days_of_cover": r["days_of_cover"], "status": r["status"],
                     "certainty_pct": r["certainty_pct"]})
    return real_backend.summarize_rows(rows, len(rows))
