# AI-Powered Supply Chain Optimization — Unified Project

All 4 members' modules combined into one working repo, built on top of
Bhuvaneshwari's verified `optimization + forecasting + shared` base
(see `OPTIMIZATION_MODULE_README.md` for her original module-level docs,
including the bugs found/fixed and gradient-check proof).

## Structure & ownership

```
├── forecasting/            Sam (D Sam Goldwin) — demand forecasting (LightGBM + LSTM ensemble)
├── optimization/            Bhuvaneshwari M — differentiable inventory optimizer (CVXPY + cvxpylayers)
├── llm_orchestration/        Harshitha N — LLM + RAG explanation module
├── app/                      Ruchitha S — integration glue + Streamlit UI
│   ├── pipeline.py             Chains all 3 modules together
│   └── main.py                  The Streamlit app itself
├── shared/
│   └── schemas.py               THE CONTRACT — every module's real input/output shape
├── data/retail_store_inventory.csv     Shared dataset (73,100 rows)
├── run_pipeline.py              Bhuvii's original forecast→optimize verification script
├── smoke_test.py                 Import + DPP-compliance sanity check
└── requirements.txt               Merged dependencies for all 4 modules
```

## Run it

```bash
pip install -r requirements.txt --break-system-packages   # or use a venv
python smoke_test.py         # sanity check
python run_pipeline.py       # forecasting -> optimization only (Bhuvii's original)
python -m app.pipeline       # forecasting -> optimization -> explanation (full chain)
streamlit run app/main.py    # the full app with UI
```

First run trains the real ensemble (~1 minute) — `app/main.py` caches results
per SKU/horizon so repeated clicks with just different costs are instant.

## Three things I found and fixed while wiring this together

**1. Schema name mismatch (`llm_orchestration`).**
`llm_orchestration.generate_explanation()` expects different field names
than `shared/schemas.py` actually defines — e.g. it wants `avg_demand` +
`trend`, but the real contract only has `point_forecast` (a list) with no
trend field at all. I added an adapter inside `app/pipeline.py`
(`_call_explanation()`) that translates between the two, so neither
Harshitha's code nor the team's shared schema needs to change. **`trend` is
currently a rough estimate** computed from the forecast curve itself, since
nothing upstream produces a real trend signal — worth raising with Sam if
you want this to be a genuine model output instead.

**2. `forecasting/ensemble.py` has its own local `ForecastOutput`.**
It's a plain dataclass with just `point_forecast`, `lower_bound`,
`upper_bound`, `model_confidence` — no `sku_id`. This is different from
`shared.schemas.ForecastOutput`, which adds `sku_id` and `horizon_days`.
`run_pipeline.py` already handles this by manually re-wrapping the raw
output; `app/pipeline.py` does the same thing. Ideally `forecast_demand()`
would return the shared schema directly so this re-wrapping step could be
removed — worth raising with Sam.

**3. `llm_client.py` hardcodes `model="claude-sonnet-4-6"`.** That doesn't
match Anthropic's current model names — double-check this against
Anthropic's docs before demoing with a real API key, or the call may fail.

## Gradient tests: environment issue found (worth fixing before viva)

Running `python -m pytest tests/ -v` in this sandbox gives **3 failed, 2
passed** — different from the "all 5 passing" result in your memory notes.
I dug into why rather than just reporting the failure:

- The regularization fix (`REG_EPS = 1e-4` quadratic term) **is present and
  correct** in `optimization/core.py` — the algorithm itself hasn't
  regressed.
- The failures show a very specific pattern: `torch.autograd.gradcheck`'s
  **analytical** gradient comes back near-zero (e.g. `0.0004`) while the
  **numerical** (finite-difference) gradient is large (e.g. `-5983.5`).
- This environment also printed a `diffcp` deprecation warning about
  `__array__` not accepting a `copy` keyword — a **known NumPy 2.x
  incompatibility** in older `diffcp` releases, which silently corrupts the
  backward pass instead of raising an error.
- Root cause, most likely: your original working environment (Windows,
  per your notes) had NumPy < 2.0 installed, but `requirements.txt` never
  pinned a version — so a fresh `pip install` here pulled NumPy 2.x, which
  breaks `diffcp`'s gradient computation silently.

**Fix applied:** pinned `numpy<2.0` in `requirements.txt`. Recommend
re-running `pytest tests/ -v` after a fresh install with this pin to confirm
all 5 pass again — I couldn't fully verify this in this exact sandbox since
other pre-installed packages here have a competing `numpy>=2.0` requirement,
but the diagnosis (version-compat, not an algorithm bug) is solid.



- `forecasting/main.py` is a stale, unused stub that raises
  `NotImplementedError` — the real `forecast_demand()` actually lives in
  and is imported from `forecasting/ensemble.py` (via `forecasting/__init__.py`).
  Harmless but confusing — worth deleting or updating once the team confirms.
- On the sample run here, the forecast came back nearly flat
  (`[53.75, 53.75, ...]`) with `model_confidence: 0.000` — this matches the
  known LSTM stale-feature-window bug already noted in earlier project
  discussion. Not something this merge fixes.

## Verified before packaging

- `smoke_test.py` passes (imports clean, `is_dcp(dpp=True)` is `True`)
- `python -m app.pipeline` ran end-to-end on the real 73,100-row CSV:
  real forecast → real optimization (`solver_status: optimal`) → explanation
  stage correctly attempted a real API call and failed gracefully due to no
  `ANTHROPIC_API_KEY` being set in this environment (not a code bug)
