# LLM Orchestration & RAG Explanation Module

**Owner:** Harshitha N (Member 3)
**Part of:** AI-Powered Supply Chain Optimization System (VVIT major project)

## What this module does

Takes the demand forecast (from Sam's module) and the optimized inventory
decision (from Bhuvii's module) and generates a plain-language,
context-grounded explanation of the recommendation — the "why" behind the
numbers.

## Folder structure

```
llm_orchestration/
├── __init__.py                  # exposes generate_explanation()
├── rag.py                       # retrieval logic (business rules / history)
├── vector_store.py              # optional embeddings-based retrieval
├── prompt_templates.py          # all LLM prompt strings, kept separate
├── llm_client.py                # thin wrapper around the LLM API
├── explainer.py                 # core orchestration logic
├── main.py                      # public entry point
├── smoke_test.py                # quick manual end-to-end check
├── requirements.txt
├── docs/
│   └── explanation_formulation.md   # design notes & interface contract
└── tests/
    └── test_explainer.py         # pytest unit tests
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here   # or set in your OS environment
```

Without an API key set, `llm_client.py` falls back to a stub response so
you can still develop and test the rest of the pipeline.

## Quick start

```bash
python smoke_test.py
```

## Usage

```python
from llm_orchestration import generate_explanation

result = generate_explanation(
    sku="S001::P0001",
    forecast_output=sam_forecast_dict,
    optimization_output=bhuvii_optimization_dict,
)

print(result["explanation"])
```

## Running tests

```bash
pytest tests/ -v
```

## TODO before this is "real"

1. Confirm exact field names in Sam's `forecast_demand()` output and
   Bhuvii's `optimize_inventory()` output — update `prompt_templates.py`
   accordingly (currently uses best-guess key names, documented in
   `docs/explanation_formulation.md`).
2. Decide on the actual knowledge base for `rag.py` (static rules vs.
   historical stats vs. real embeddings via `vector_store.py`).
3. Get a real API key and test with `llm_client.py` actually calling the
   LLM (not just the stub).
4. Confirm the output JSON shape (`sku`, `explanation`, `sources`) against
   what Ruchitha's Streamlit UI expects to render.
