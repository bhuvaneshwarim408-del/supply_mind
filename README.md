# Integration + Flask UI — Ruchitha S

This repo isolates **Ruchitha's individual contribution** (`app/`) for
submission and viva prep, following the same pattern as Bhuvaneshwari's
`optimization/` submission repo. `forecasting/`, `optimization/`,
`llm_orchestration/`, and `shared/` are included **unmodified**, only so
the full pipeline runs end-to-end for the demo.

## Your files

```
app/
├── __init__.py
├── pipeline.py         # Chains forecasting -> optimization -> explanation
├── main.py               # The Flask app itself (routes, form handling)
├── templates/
│   └── index.html          # The dashboard page (Jinja2 template)
└── static/
    └── style.css             # Styling
```

## What it does

Given a SKU and a set of business constraints (warehouse capacity, budget,
lead time, costs), the app:
1. Calls Sam's real `forecast_demand()` on the real CSV
2. Feeds that into Bhuvaneshwari's real `optimize_inventory()`
3. Feeds both into Harshitha's `generate_explanation()` for a plain-language
   summary
4. Displays all of it on a single dashboard page: a forecast table
   (day-by-day, with confidence bounds), three key metrics (order quantity
   / safety stock / reorder point), total expected cost, and the
   explanation text with its sources

## Run it

```bash
pip install -r requirements.txt --break-system-packages   # or use a venv
python -m app.main
```
Then open **http://127.0.0.1:5000** in a browser.

First run trains the real ensemble (~1 min) — results are cached in memory
per (SKU, horizon, constraints) combination afterward, so re-running with
the same inputs is instant. Restarting the Flask server clears this cache
(it's a plain Python dict, not persisted to disk).

You can also test the integration layer without the UI:
```bash
python -m app.pipeline
```

## Design notes for your understanding / viva

- **`app/pipeline.py` never raises.** Every stage (forecast, optimize,
  explain) is wrapped in its own `try/except`, and the function always
  returns a dict with an `"error"` key set instead of crashing — so a
  failure in one stage (e.g. a missing API key for the explanation stage)
  still lets the page show whatever DID succeed.
- **No adapter needed for the explanation stage.** Harshitha's
  `generate_explanation()` accepts `shared.schemas.ForecastOutput` /
  `OptimizationOutput` objects directly, so `app/pipeline.py` just passes
  them straight through — no field-name translation needed.
- **Why re-wrap the forecast output:** `forecasting/ensemble.py` returns
  its own small dataclass (no `sku_id`), different from
  `shared.schemas.ForecastOutput` (which has `sku_id` + `horizon_days`).
  `pipeline.py` manually re-wraps it into the shared schema — the same
  pattern already used in the team's `run_pipeline.py`.
- **Caching:** training the real ensemble takes ~50 seconds, so
  `app/main.py` keeps a simple in-memory dict (`_pipeline_cache`) keyed on
  every input field, so identical form submissions skip retraining.
- **Why Flask over Streamlit:** Flask gives full control over the HTML/CSS
  (useful if the panel asks about the UI layer specifically), and the
  request/response model (GET shows the form, POST runs the pipeline and
  re-renders the same template with results) is a good one to be able to
  explain in viva — it's the same pattern most web apps use.

## Verified before packaging

Tested using Flask's test client (`app.test_client()`), not just a manual
browser check:
- `GET /` returns 200 and renders the empty form correctly
- `POST /` with real form data ran the full pipeline end-to-end on the
  real 73,100-row CSV: real forecast → real optimization
  (`order_quantity: 24.9`, matching the value verified independently via
  `run_pipeline.py`) → explanation stage rendered the free rule-based
  template fallback correctly (no `GROQ_API_KEY` set in this test
  environment — with a real key, the same section shows live Groq output
  instead, no code change needed)
- Forecast table, stat cards, and sources list all confirmed present in
  the rendered HTML
