import json
import os
import time
from collections import defaultdict, deque
from functools import wraps

from flask import Blueprint, Response, abort, current_app, flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from .auth import safe_next
from .db import get_db
from .services import data_source

bp = Blueprint("views", __name__)


def viewer(fn):
    """Pages and read-only APIs: open to everyone when SUPPLYMIND_PUBLIC=1, otherwise they need a login."""
    @wraps(fn)
    def inner(*args, **kwargs):
        if current_app.config.get("PUBLIC_VIEW"):
            return fn(*args, **kwargs)
        return login_required(fn)(*args, **kwargs)
    return inner


_hits = defaultdict(deque)


def rate_limit(name, max_calls=20, window=60):
    """Tiny in-memory limiter for compute-heavy endpoints. Each endpoint has its own counter per client IP."""
    now = time.time()
    q = _hits[(name, request.remote_addr or "?")]
    while q and now - q[0] > window:
        q.popleft()
    if len(q) >= max_calls:
        abort(429, "Too many requests for this feature. Wait a minute and try again.")
    q.append(now)

HORIZONS = (7, 14, 30)
# name: (type, min, max)
CONSTRAINT_RULES = {
    "warehouse_capacity": (float, 1, 1e7),
    "budget": (float, 1, 1e9),
    "lead_time_days": (int, 0, 90),
    "unit_holding_cost": (float, 0, 1e6),
    "unit_stockout_penalty": (float, 0, 1e6),
    "min_order_qty": (float, 0, 1e7),
}
LABELS = {
    "warehouse_capacity": "Warehouse capacity (units)",
    "budget": "Budget",
    "lead_time_days": "Supplier lead time (days)",
    "unit_holding_cost": "Holding cost per unit",
    "unit_stockout_penalty": "Stockout penalty per unit",
    "min_order_qty": "Minimum order quantity (units)",
}


def parse_params(args):
    """Return (horizon, constraints). Raises ValueError with a message the user can act on."""
    try:
        horizon = int(args.get("horizon", data_source.DEFAULT_HORIZON))
    except ValueError:
        raise ValueError("Forecast horizon must be 7, 14 or 30 days.")
    if horizon not in HORIZONS:
        raise ValueError("Forecast horizon must be 7, 14 or 30 days.")
    constraints = dict(data_source.DEFAULT_CONSTRAINTS)
    for name, (kind, lo, hi) in CONSTRAINT_RULES.items():
        raw = args.get(name)
        if raw in (None, ""):
            continue
        try:
            value = kind(raw)
        except ValueError:
            raise ValueError(f"{LABELS[name]} must be a number.")
        if not (lo <= value <= hi) or value != value:
            raise ValueError(f"{LABELS[name]} must be between {lo:g} and {hi:g}.")
        constraints[name] = value
    return horizon, constraints


def _pick_sku():
    sku = request.args.get("sku", data_source.SKUS[0])
    return sku if sku in data_source.SKUS else data_source.SKUS[0]


def _page_params():
    try:
        return parse_params(request.args)
    except ValueError as exc:
        flash(str(exc) + " Showing the defaults instead.", "error")
        return data_source.DEFAULT_HORIZON, dict(data_source.DEFAULT_CONSTRAINTS)


@bp.route("/")
@viewer
def dashboard():
    return render_template("dashboard.html")


@bp.route("/forecast")
@viewer
def forecast():
    horizon, _ = _page_params()
    return render_template("forecast.html", skus=data_source.SKUS, sku=_pick_sku(), horizon=horizon, horizons=HORIZONS)


@bp.route("/optimizer")
@viewer
def optimizer():
    horizon, constraints = _page_params()
    sku = _pick_sku()
    recent = get_db().execute(
        "SELECT o.*, u.name AS user_name FROM overrides o JOIN users u ON u.id = o.user_id "
        "WHERE o.sku = ? ORDER BY o.id DESC LIMIT 5", (sku,)).fetchall()
    return render_template(
        "optimizer.html", skus=data_source.SKUS, sku=sku, horizon=horizon, horizons=HORIZONS,
        constraints=constraints, labels=LABELS, overrides=recent,
    )


@bp.route("/overrides", methods=["POST"])
@login_required
def add_override():
    sku = request.form.get("sku", "")
    back = safe_next(request.form.get("next")) or url_for("views.optimizer")
    try:
        recommended = float(request.form.get("recommended_qty", ""))
        chosen = float(request.form.get("chosen_qty", ""))
    except ValueError:
        flash("Enter the quantity as a number, and reload the page if the recommended amount is missing.", "error")
        return redirect(back)
    reason = request.form.get("reason", "").strip()
    if sku not in data_source.SKUS:
        flash("Unknown SKU.", "error")
    elif not (0 <= chosen <= 1e7) or recommended < 0:
        flash("The quantity must be between 0 and 10,000,000.", "error")
    elif not reason:
        flash("Add a short reason so the change can be reviewed later.", "error")
    else:
        db = get_db()
        db.execute("INSERT INTO overrides (user_id, sku, recommended_qty, chosen_qty, reason) VALUES (?, ?, ?, ?, ?)",
                   (current_user.id, sku, recommended, chosen, reason[:200]))
        db.commit()
        flash("Saved to the change log.", "info")
    return redirect(back)


@bp.route("/whatif")
@viewer
def whatif():
    from scm import whatif as wi
    return render_template("whatif.html", skus=wi.sku_list())


@bp.route("/alerts")
@viewer
def alerts():
    return render_template("alerts.html")


@bp.route("/api/alerts")
@viewer
def api_alerts():
    return jsonify(data_source.portfolio()["alerts"])


def _load_experiments():
    path = current_app.config.get("EXPERIMENTS_PATH", "results/experiments.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def _pdf(data, name):
    return Response(data, mimetype="application/pdf", headers={"Content-Disposition": f'attachment; filename="{name}"'})


@bp.route("/report.pdf")
@viewer
def report_pdf():
    from datetime import date
    from . import report
    rate_limit("report", 10)
    return _pdf(report.portfolio_report(data_source.portfolio(), _load_experiments(), data_source.IS_DEMO),
                f"supplymind-report-{date.today().isoformat()}.pdf")


@bp.route("/report/<sku>.pdf")
@viewer
def report_sku_pdf(sku):
    from datetime import date
    from . import report
    if sku not in data_source.SKUS:
        abort(404)
    rate_limit("report", 10)
    horizon, constraints = _api_params()
    fc = data_source.forecast(sku, horizon)
    plan = data_source.optimizer(sku, horizon, constraints, generate=False)
    return _pdf(report.sku_report(fc, plan, data_source.IS_DEMO), f"supplymind-{sku.replace(':', '_')}-{date.today().isoformat()}.pdf")


@bp.route("/rl")
@viewer
def rl_lab():
    return render_template("rl.html")


@bp.route("/experiments")
@viewer
def experiments():
    return render_template("experiments.html")


@bp.route("/api/experiments")
@viewer
def api_experiments():
    path = current_app.config.get("EXPERIMENTS_PATH", "results/experiments.json")
    if not os.path.exists(path):
        abort(404, "No experiment results yet. Run: python scripts/run_experiments.py")
    with open(path, encoding="utf-8") as fh:
        return jsonify(json.load(fh))


@bp.route("/api/whatif/<sku>")
@viewer
def api_whatif(sku):
    from scm import whatif as wi
    if sku not in wi.sku_list():
        abort(404)
    rate_limit("whatif")
    try:
        change = float(request.args.get("demand_change", 0))
        delay = int(request.args.get("delay", 0))
        cap_raw = request.args.get("capacity", "")
        cap = float(cap_raw) if cap_raw not in ("", None) else None
    except ValueError:
        abort(400, "Demand change, delay and storage limit must be numbers.")
    if not (-80 <= change <= 300):
        abort(400, "Demand change must be between -80 and 300 percent.")
    if not (0 <= delay <= 14):
        abort(400, "Supplier delay must be between 0 and 14 days.")
    if cap is not None and not (1 <= cap <= 1e7):
        abort(400, "Storage limit must be at least 1 unit, or left blank for no limit.")
    return jsonify(wi.run(sku, change, delay, cap))


def _api_params():
    try:
        return parse_params(request.args)
    except ValueError as exc:
        abort(400, str(exc))


@bp.route("/api/dashboard")
@viewer
def api_dashboard():
    return jsonify(data_source.portfolio())


@bp.route("/api/forecast/<sku>")
@viewer
def api_forecast(sku):
    if sku not in data_source.SKUS:
        abort(404)
    horizon, _ = _api_params()
    return jsonify(data_source.forecast(sku, horizon))


@bp.route("/api/optimizer/<sku>")
@viewer
def api_optimizer(sku):
    if sku not in data_source.SKUS:
        abort(404)
    horizon, constraints = _api_params()
    return jsonify(data_source.optimizer(sku, horizon, constraints, generate=current_user.is_authenticated))
