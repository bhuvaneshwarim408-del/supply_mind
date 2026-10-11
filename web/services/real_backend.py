"""Glue between the web app and the team's real modules.

Calls forecasting.forecast_demand, optimization.optimize_inventory and
llm_orchestration.generate_explanation exactly as app/pipeline.py does.
Training a forecast takes about a minute per SKU, so results are cached as
JSON files in instance/cache (delete that folder to force fresh runs).
"""
import hashlib
import json
import os
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from . import policy

CSV_PATH = os.environ.get("SUPPLYMIND_CSV", "data/retail_store_inventory.csv")
CACHE_DIR = Path(os.environ.get("SUPPLYMIND_CACHE", "instance/cache"))
DEFAULT_HORIZON = 14
DEFAULT_CONSTRAINTS = {
    "warehouse_capacity": 500.0,
    "budget": 5000.0,
    "lead_time_days": 7,
    "unit_holding_cost": 2.0,
    "unit_stockout_penalty": 10.0,
    "min_order_qty": 20.0,
}

_train_lock = threading.Lock()
_modules = None
_df = None
_cache_only = False


class NotAnalysed(Exception):
    """Raised in cache-only mode (no PyTorch on this server) for a SKU that was not pre-computed."""


def _load_modules():
    global _modules
    if _modules is None:
        from forecasting import forecast_demand, set_data_source
        from forecasting.data import load_csv
        from optimization import optimize_inventory
        from shared.schemas import ForecastOutput, OptimizationConstraints, OptimizationOutput

        _modules = {
            "forecast_demand": forecast_demand,
            "set_data_source": set_data_source,
            "load_csv": load_csv,
            "optimize_inventory": optimize_inventory,
            "ForecastOutput": ForecastOutput,
            "OptimizationConstraints": OptimizationConstraints,
            "OptimizationOutput": OptimizationOutput,
        }
    return _modules


def available():
    """True when real results can be served: full models, or (cache-only) pre-computed results on disk."""
    global _cache_only
    if os.environ.get("SUPPLYMIND_DEMO") == "1" or not Path(CSV_PATH).exists():
        return False
    try:
        _load_modules()
        _cache_only = False
        return True
    except Exception:
        pass
    if any(CACHE_DIR.glob("plan_*.json")):
        _cache_only = True
        return True
    return False


def cache_only():
    return _cache_only


def _load_csv_plain(path):
    import pandas as pd
    df = pd.read_csv(path, parse_dates=["Date"])
    df["sku_id"] = df["Store ID"] + "::" + df["Product ID"]
    return df.sort_values(["sku_id", "Date"]).reset_index(drop=True)


def _frame():
    global _df
    if _df is None:
        if _cache_only:
            _df = _load_csv_plain(CSV_PATH)
        else:
            m = _load_modules()
            _df = m["load_csv"](CSV_PATH)
            m["set_data_source"](_df)
    return _df


def _default_plans():
    out = []
    for path in sorted(CACHE_DIR.glob("plan_*.json")):
        plan = _read(path)
        if plan and plan.get("horizon") == DEFAULT_HORIZON and plan.get("constraints") == DEFAULT_CONSTRAINTS:
            out.append(plan)
    return out


def skus():
    if _cache_only:
        return sorted({p["sku"] for p in _default_plans()})
    return sorted(_frame()["sku_id"].unique())


def history(sku, n=45):
    df = _frame()
    rows = df[df["sku_id"] == sku].tail(n)
    return [d.date().isoformat() for d in rows["Date"]], [int(round(v)) for v in rows["Units Sold"]]


def stock_now(sku):
    df = _frame()
    return float(df[df["sku_id"] == sku]["Inventory Level"].iloc[-1])


# ---------- cache ----------
def _key(*parts):
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def _read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def get_forecast(sku, horizon):
    """Trained forecast as a shared.schemas.ForecastOutput dict (cached)."""
    path = CACHE_DIR / f"fc_{_key(sku, horizon)}.json"
    cached = _read(path)
    if cached:
        return cached
    if _cache_only:
        raise NotAnalysed(sku)
    with _train_lock:
        cached = _read(path)
        if cached:
            return cached
        m = _load_modules()
        _frame()
        raw = m["forecast_demand"](sku, horizon)
        fc = m["ForecastOutput"](  # re-wrap into the shared contract, as app/pipeline.py does
            sku_id=sku, horizon_days=horizon,
            point_forecast=list(raw.point_forecast), lower_bound=list(raw.lower_bound),
            upper_bound=list(raw.upper_bound), model_confidence=raw.model_confidence,
        ).model_dump()
        _write(path, fc)
        return fc


def get_plan(sku, horizon, constraints):
    """Forecast + optimizer result for one SKU and one set of constraints (cached)."""
    path = CACHE_DIR / f"plan_{_key(sku, horizon, constraints)}.json"
    cached = _read(path)
    if cached:
        return cached
    if _cache_only:
        raise NotAnalysed(sku)
    m = _load_modules()
    fc = get_forecast(sku, horizon)
    opt = m["optimize_inventory"](m["ForecastOutput"](**fc), m["OptimizationConstraints"](**constraints))
    plan = {
        "sku": sku, "horizon": horizon, "constraints": constraints,
        "forecast": fc, "optimization": opt.model_dump(), "explanation": None,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    _write(path, plan)
    return plan


def _allowed_numbers(plan):
    fc, opt, c = plan["forecast"], plan["optimization"], plan["constraints"]
    vals = [opt["order_quantity"], opt["safety_stock"], opt["reorder_point"], opt["total_expected_cost"],
            100 * fc["model_confidence"], fc["horizon_days"], sum(fc["point_forecast"]), *c.values(),
            *fc["point_forecast"], *fc["lower_bound"], *fc["upper_bound"]]
    pf = fc["point_forecast"]
    vals += [sum(pf) / len(pf), min(pf), max(pf)]
    try:
        vals.append(stock_now(plan["sku"]))
    except Exception:
        pass
    return vals


def _with_check(plan, ex):
    from scm import faithfulness
    out = dict(ex)
    out["check"] = faithfulness.check(ex["explanation"], _allowed_numbers(plan)) if ex.get("explanation") else None
    return out


def explain(plan, generate=True):
    """Plain-language explanation from the LLM module. Only successful text is cached."""
    saved = plan.get("explanation")
    if saved and saved.get("explanation"):
        return _with_check(plan, saved)
    if not generate or _cache_only:
        return None
    try:
        from llm_orchestration import generate_explanation
    except ImportError:
        return None
    m = _load_modules()
    try:
        out = generate_explanation(
            plan["sku"], m["ForecastOutput"](**plan["forecast"]), m["OptimizationOutput"](**plan["optimization"])
        )
    except Exception as exc:
        return {"explanation": None, "sources": [], "error": str(exc)}
    if out and out.get("explanation"):
        plan["explanation"] = {"explanation": out["explanation"], "sources": [str(s) for s in out.get("sources", [])]}
        path = CACHE_DIR / f"plan_{_key(plan['sku'], plan['horizon'], plan['constraints'])}.json"
        _write(path, plan)
        return _with_check(plan, plan["explanation"])
    return {"explanation": None, "sources": [], "error": "The explanation module returned no text."}


# ---------- views (same shapes as the demo data) ----------
def _future_dates(last_iso, n):
    last = date.fromisoformat(last_iso)
    return [(last + timedelta(days=i + 1)).isoformat() for i in range(n)]


def _status(stock, rop):
    if stock <= rop:
        return "Reorder now"
    return "Low" if stock <= rop * 1.3 else "Healthy"


def forecast_view(sku, horizon):
    fc = get_forecast(sku, horizon)
    hd, hv = history(sku)
    n = len(hd)
    pt, lo, hi = fc["point_forecast"], fc["lower_bound"], fc["upper_bound"]
    mean = sum(pt) / len(pt)
    half_range = sum((u - l) / 2 for l, u in zip(lo, hi)) / len(pt)
    pad = [None] * n
    return {
        "sku": sku,
        "dates": hd + _future_dates(hd[-1], len(pt)),
        "actual": hv + [None] * len(pt),
        "forecast": pad + [round(v, 1) for v in pt],
        "lower": pad + [round(v, 1) for v in lo],
        "upper": pad + [round(v, 1) for v in hi],
        "split_index": n,
        "metrics": {
            "range_pct": round(100 * half_range / mean, 1) if mean > 0 else 0.0,
            "certainty_pct": round(max(0.0, 100 - 100 * half_range / mean), 1) if mean > 0 else 0.0,
            "lower_avg": round(sum(lo) / len(lo)), "upper_avg": round(sum(hi) / len(hi)),
            "confidence_pct": round(100 * fc["model_confidence"], 1),
            "horizon_total": round(sum(pt)),
            "avg_daily": round(mean),
            "horizon_days": len(pt),
        },
    }


def _plan_summary(plan):
    fc, opt, c = plan["forecast"], plan["optimization"], plan["constraints"]
    pt = fc["point_forecast"]
    mean = sum(pt) / len(pt)
    half = sum((u - l) / 2 for l, u in zip(fc["lower_bound"], fc["upper_bound"])) / len(pt)
    now = stock_now(plan["sku"])
    return {
        "certainty_pct": round(max(0.0, 100 - 100 * half / mean), 1) if mean > 0 else 0.0,
        "sku": plan["sku"], "mean": mean, "stock_now": now,
        "days_of_cover": round(now / mean, 1) if mean > 0 else 0.0,
        "status": _status(now, opt["reorder_point"]),
        "confidence_pct": round(100 * fc["model_confidence"], 1),
        "lead": c["lead_time_days"],
    }


def optimizer_view(sku, horizon, constraints, with_explanation=True, generate=True):
    plan = get_plan(sku, horizon, constraints)
    fc, opt = plan["forecast"], plan["optimization"]
    s = _plan_summary(plan)
    pt = fc["point_forecast"]
    sim = policy.simulate(s["stock_now"], pt, opt["reorder_point"], opt["order_quantity"], constraints["lead_time_days"])
    hd, _ = history(sku, 1)
    expl = explain(plan, generate) if with_explanation else None
    return {
        "sku": sku,
        "projected": True,
        "dates": _future_dates(hd[-1], len(pt)),
        "stock": sim["stock"],
        "reorder_point": opt["reorder_point"],
        "safety_stock": opt["safety_stock"],
        "order_qty": opt["order_quantity"],
        "lead_time_days": constraints["lead_time_days"],
        "orders_placed": sim["orders"],
        "projected_service_pct": sim["service_pct"],
        "total_cost": opt["total_expected_cost"],
        "solver_status": opt["solver_status"],
        "confidence_pct": s["confidence_pct"],
        "certainty_pct": s["certainty_pct"],
        "avg_daily": round(s["mean"]),
        "horizon_days": len(pt),
        "stock_now": s["stock_now"],
        "days_of_cover": s["days_of_cover"],
        "status": s["status"],
        "explanation": expl,
    }


def portfolio_view():
    """Summary of SKUs already analysed with the default settings."""
    sku_list = skus()
    rows = []
    for plan in _default_plans():
        sm = _plan_summary(plan)
        opt = plan["optimization"]
        rows.append({**sm, "cost": opt["total_expected_cost"], "qty": opt["order_quantity"], "rop": opt["reorder_point"]})
    return summarize_rows(rows, len(sku_list))


def summarize_rows(rows, sku_count):
    n = len(rows)
    counts = {"Reorder now": 0, "Low": 0, "Healthy": 0, "On order": 0}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    needing = [r for r in rows if r["status"] == "Reorder now"]
    return {
        "kpis": {
            "analysed": n,
            "sku_count": sku_count,
            "total_cost": round(sum(r["cost"] for r in rows)),
            "avg_certainty_pct": round(sum(r["certainty_pct"] for r in rows) / n, 1) if n else 0.0,
            "reorder_now": counts["Reorder now"],
            "reorder_units": round(sum(r["qty"] for r in needing)),
        },
        "status_counts": counts,
        "cost_by_sku": [{"sku": r["sku"], "cost": r["cost"]} for r in sorted(rows, key=lambda r: -r["cost"])[:10]],
        "top_orders": [{"sku": r["sku"], "qty": r["qty"]} for r in sorted(rows, key=lambda r: -r["qty"])[:10]],
        "at_risk": [
            {"sku": r["sku"], "stock_now": r["stock_now"], "reorder_point": r["rop"],
             "days_of_cover": r["days_of_cover"], "status": r["status"]}
            for r in sorted(rows, key=lambda r: r["days_of_cover"])[:8]
        ],
        "alerts": [
            {"sku": r["sku"], "status": r["status"], "stock_now": r["stock_now"], "reorder_point": r["rop"],
             "days_of_cover": r["days_of_cover"], "order_qty": r["qty"], "certainty_pct": r["certainty_pct"]}
            for r in sorted(rows, key=lambda r: r["days_of_cover"])
        ],
    }
