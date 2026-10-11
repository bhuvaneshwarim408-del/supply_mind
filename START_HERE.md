# SupplyMind: AI for supply chain

Demand forecasting with upper and lower bounds, inventory optimization, simulation, reinforcement learning, experiments and a website, in one folder.

## What is where

| Folder / file | What it does |
|---|---|
| `forecasting/`, `optimization/`, `llm_orchestration/`, `shared/`, `app/` | The team's original modules (LightGBM + LSTM forecaster, differentiable CVXPY optimizer, LLM explanations, first Flask UI). Unchanged. |
| `scm/` | Decision core (numpy, pandas, CVXPY only): data, features, forecasters with conformal ranges, decision-focused model, inventory simulator, policies, RL agent, backtests, what-if, explanation number check. |
| `scripts/run_experiments.py` | Runs every experiment and writes `results/experiments.json`. |
| `results/experiments.json` | Results already generated, so Experiments and RL Lab work immediately. |
| `web/` | The SupplyMind website. |
| `Dockerfile`, `Procfile`, `render.yaml`, `requirements-web.txt` | For putting the site online. |
| `web/report.py` | Builds the PDF reports (portfolio report and one-SKU plan). |
| `tests/test_scm.py`, `tests/test_web.py` | 39 tests for the new code. |

## Website pages

Dashboard (cost, units to reorder, urgent/warning/healthy tiles, status donut, **Download PDF report**) · Alerts Center · Demand Forecast (chart with lower and upper bound lines, plus a daily bounds table) · Smart Reorder Planner (order quantity, reorder point, safety stock, projected stock, AI explanation with number check, order change log, **Download PDF plan**) · What-if Simulator · RL Lab · Experiments. Dark theme by default, with a light option.

## Run it on your laptop (Windows PowerShell, from this folder)

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    $env:SECRET_KEY = "put-a-long-random-string-here"
    python run_web.py

Open http://127.0.0.1:5001 and create an account. Light install (no PyTorch): `pip install flask flask-login numpy pandas cvxpy reportlab pypdf pytest`. Then Dashboard, Forecast and Planner show demo data, while What-if, Experiments and RL Lab work fully.

To use real forecasts, train some SKUs once (about a minute each):

    python -m web.precompute --n 20 --explain

The result is saved in `instance/cache/` and the pages then load instantly.

## Put it online so anyone can open it

A free host cannot train PyTorch models, so the public site shows results you pre-computed on your laptop.

1. Run `python -m web.precompute --n 20 --explain` on your laptop (use more SKUs if you have time).
2. Put this whole folder, including `instance/cache/`, in a GitHub repository.
3. On Render (or Railway), create a Web Service from the repo. `render.yaml` and the `Dockerfile` are already set up.
4. Set the environment variables: `SECRET_KEY` (any long random text), `SUPPLYMIND_PUBLIC=1` (anyone can view; only signed-in users can write), and `SUPPLYMIND_ADMIN_EMAIL` (your email, so a stranger cannot become the admin).

Things to know: on free plans the user database resets when the site restarts, and the site sleeps when idle. Public visitors can view cached SKUs, but only signed-in users can trigger AI explanations. The What-if page is limited to 20 runs per minute per visitor.

## PDF reports

- **Dashboard, Download PDF report:** summary figures, the SKUs that need action, the largest orders, and the experiment results.
- **Demand Forecast or Smart Reorder Planner, Download PDF plan:** one SKU with its numbers, the forecast chart with lower and upper bounds, the daily bounds table, projected stock, and the AI explanation if one was already generated.
- Reports made from demo data carry a red DEMO DATA banner. Reports never call the AI service, so public visitors cannot use it up.

## Re-run the experiments

    python scripts/run_experiments.py --skus 60 --episodes 12000     # about 3 minutes
    python scripts/run_experiments.py --quick                        # about 20 seconds

## Tests

    python -m pytest tests/test_scm.py tests/test_web.py -v

## How the experiments are set up

- **Time split:** train on the first 70% of days, tune on the next 15%, report on the last 15%, never shuffled.
- **Simulator:** orders arrive after a 3-day lead time, demand is served from stock, unmet demand is lost. Cost is holding plus stockout cost, scaled by each SKU's relative price.
- **Policies:** simple rule; forecast base-stock (with a conformal safety margin); the same with a decision-focused model; a CVXPY scenario optimizer; an RL correction.
- **Fairness:** the service level is chosen on validation days only. Two storage cases: ample, and tight (85% of normal).
- **Three datasets:** the real CSV, a synthetic set with patterns, and the same with volatile promotions. Synthetic sets are labelled as such.

## What the results say (honest summary)

| Cost vs the simple rule, test days | Real CSV | Synthetic | Volatile promos |
|---|---|---|---|
| Forecast-based policy, ample storage | -29% | -60% | -50% |
| Same, tight storage | no gain | -12% | -6% |

- A calibrated forecast with the right safety stock beats the simple reorder rule by a large margin. The conformal 80% range covers about 80% of actual demand on the real data.
- The real CSV has almost no pattern beyond each SKU's average, so savings come from the safety stock, not clever forecasting.
- The CVXPY optimizer equals the forecast policy when storage is ample (a test confirms it matches the textbook newsvendor answer). With tight shared storage it was worse than simple proportional scaling. It does report shadow prices.
- Decision-focused training was within 1-2% of the squared-error model, not clearly better.
- The RL correction did not beat the forecast policy (RL Lab shows the learning curve and what it chose).

Use these numbers as they are. They are real outputs of this code.

## About "forecast certainty"

The old module's `model_confidence` is 1 minus MAPE, and this dataset's MAPE is above 100%, so it shows 0.0%. The website shows **forecast certainty** instead: 100% minus the typical plus-or-minus range of the forecast (lower to upper bound). It is defined on the Forecast page.

## Known limits

- Experiments use light forecasters so they run anywhere. The LightGBM + LSTM ensemble powers the live single-SKU pages but is not in the backtest.
- User roles (admin and planner) are labels for now; nothing is restricted by role yet.
- Costs are in relative units, because the CSV does not give a currency.
- No login rate limiting or password reset yet.
- Not included: robust optimization, multi-supplier ordering, drift monitoring.
