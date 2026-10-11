"""
app/main.py
Owner: Ruchitha S -- Flask UI

Run from the project root with:
    python -m app.main

Then open http://127.0.0.1:5000 in a browser.

Uses the REAL forecasting -> optimization -> explanation pipeline
(app/pipeline.py), built on the team's shared/schemas.py contracts.
"""

from flask import Flask, render_template, request

from shared.schemas import OptimizationConstraints
from app.pipeline import run_pipeline, EXPLANATION_MODULE_AVAILABLE

app = Flask(__name__)

# Simple in-memory cache: training the real ensemble takes ~50s, so we
# avoid retraining every time someone reloads the page with the same
# inputs. Keyed on (sku_id, horizon_days, constraints tuple).
_pipeline_cache = {}


def _cached_run(sku_id, horizon_days, constraints: OptimizationConstraints):
    cache_key = (
        sku_id, horizon_days, constraints.warehouse_capacity, constraints.budget,
        constraints.lead_time_days, constraints.unit_holding_cost,
        constraints.unit_stockout_penalty, constraints.min_order_qty,
    )
    if cache_key not in _pipeline_cache:
        _pipeline_cache[cache_key] = run_pipeline(
            sku_id=sku_id, horizon_days=horizon_days, constraints=constraints
        )
    return _pipeline_cache[cache_key]


DEFAULTS = {
    "sku_id": "S001::P0001",
    "horizon_days": 14,
    "warehouse_capacity": 500.0,
    "budget": 5000.0,
    "lead_time_days": 7,
    "unit_holding_cost": 2.0,
    "unit_stockout_penalty": 10.0,
    "min_order_qty": 20.0,
}


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    form_values = DEFAULTS.copy()

    if request.method == "POST":
        # Pull form inputs, falling back to defaults if something's missing.
        form_values["sku_id"] = request.form.get("sku_id", DEFAULTS["sku_id"])
        form_values["horizon_days"] = int(request.form.get("horizon_days", DEFAULTS["horizon_days"]))
        form_values["warehouse_capacity"] = float(request.form.get("warehouse_capacity", DEFAULTS["warehouse_capacity"]))
        form_values["budget"] = float(request.form.get("budget", DEFAULTS["budget"]))
        form_values["lead_time_days"] = int(request.form.get("lead_time_days", DEFAULTS["lead_time_days"]))
        form_values["unit_holding_cost"] = float(request.form.get("unit_holding_cost", DEFAULTS["unit_holding_cost"]))
        form_values["unit_stockout_penalty"] = float(request.form.get("unit_stockout_penalty", DEFAULTS["unit_stockout_penalty"]))
        form_values["min_order_qty"] = float(request.form.get("min_order_qty", DEFAULTS["min_order_qty"]))

        constraints = OptimizationConstraints(
            warehouse_capacity=form_values["warehouse_capacity"],
            budget=form_values["budget"],
            lead_time_days=form_values["lead_time_days"],
            unit_holding_cost=form_values["unit_holding_cost"],
            unit_stockout_penalty=form_values["unit_stockout_penalty"],
            min_order_qty=form_values["min_order_qty"],
        )

        result = _cached_run(form_values["sku_id"], form_values["horizon_days"], constraints)

    return render_template(
        "index.html",
        form_values=form_values,
        result=result,
        explanation_available=EXPLANATION_MODULE_AVAILABLE,
    )


if __name__ == "__main__":
    app.run(debug=True)
