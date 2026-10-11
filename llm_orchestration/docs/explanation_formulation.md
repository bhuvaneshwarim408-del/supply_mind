# LLM Orchestration & RAG Explanation Module — Design Notes

**Owner:** Harshitha N (Member 3)

## Purpose

This module sits at the end of the pipeline. It takes the numeric outputs
from the forecasting module (Sam) and the optimization module (Bhuvii) and
converts them into a plain-language explanation that a non-technical user
(store manager, viva panel) can understand.

```
Forecast (Sam) ──┐
                  ├──► Explanation module (this) ──► Streamlit UI (Ruchitha)
Optimization (Bhuvii) ──┘
```

## Pipeline stages

1. **Retrieval (rag.py)** — gather supporting context relevant to the
   current SKU/decision (business rules, historical notes, or embedding-based
   similarity search via vector_store.py).
2. **Prompt construction (prompt_templates.py)** — combine the forecast
   numbers, optimization numbers, and retrieved context into a structured
   prompt.
3. **LLM call (llm_client.py)** — send the prompt to the LLM and get back a
   natural-language explanation.
4. **Orchestration (explainer.py / main.py)** — wire the above steps
   together and expose a single clean function, `generate_explanation()`,
   for Ruchitha's UI to call.

## Design decisions to make (and be ready to defend in viva)

- **Why RAG instead of just prompting the LLM with raw numbers?**
  Grounding the explanation in retrieved context (business rules / history)
  reduces hallucination and makes the explanation traceable back to a
  source, rather than the LLM inventing justifications.
- **What is the knowledge base?** Decide early: static rules file vs.
  historical CSV stats vs. real embeddings. Static rules are simplest and
  perfectly defensible for this project's scope; only add embeddings if
  time allows, since it adds real complexity for arguably little extra
  value at this scale.
- **Why keep prompts in a separate file (prompt_templates.py)?** Makes
  prompt engineering visible and iterable without touching orchestration
  logic — useful to point to during a viva as evidence of deliberate
  design rather than ad hoc prompting.
- **Failure handling:** what happens if the LLM API call fails or times
  out? Right now `llm_client.py` has a stub fallback so the pipeline never
  hard-crashes; consider whether a rule-based fallback explanation (e.g. a
  templated sentence built directly from the numbers) is worth adding as a
  safety net, mirroring how Bhuvii's module has a `fallback.py`.

## Interface contract (confirm exact field names against real modules)

**Input from Sam (forecast_output):**
```json
{
  "horizon": 14,
  "avg_demand": 45.2,
  "trend": "rising"
}
```

**Input from Bhuvii (optimization_output):**
```json
{
  "order_quantity": 320.0,
  "safety_stock": 85.0,
  "reorder_point": 210.0,
  "solver_status": "optimal",
  "total_cost": 1450.75
}
```

**Output to Ruchitha (UI):**
```json
{
  "sku": "S001::P0001",
  "explanation": "Based on a forecasted demand of 45.2 units/day with a rising trend, we recommend ordering 320 units...",
  "sources": ["Business rule: ...", "Historical note: ..."]
}
```

⚠️ These field names are best guesses based on the pattern used elsewhere
in the project. Confirm the exact keys Sam's `forecast_demand()` and
Bhuvii's `optimize_inventory()` actually return, and update
`prompt_templates.py` / `explainer.py` accordingly before integration.
