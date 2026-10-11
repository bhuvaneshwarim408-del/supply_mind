"""
explainer.py
------------
Core orchestration logic:
    forecast_output + optimization_output
        -> retrieve context (rag.py)
        -> build prompt (prompt_templates.py)
        -> call LLM (llm_client.py)
        -> return structured explanation

This is the module-level equivalent of your (Bhuvii's) core.py / Sam's
forecast engine - the "brain" that the outward-facing main.py wraps.
"""

from typing import Dict, Any

from shared.schemas import ForecastOutput, OptimizationOutput
from .rag import retrieve_context
from .prompt_templates import build_explanation_prompt
from .llm_client import LLMClient

_llm_client = LLMClient()


def explain_decision(sku: str, forecast: ForecastOutput,
                      optimization: OptimizationOutput) -> Dict[str, Any]:
    """
    Produce a structured explanation for one SKU's inventory decision.

    Returns
    -------
    dict with keys:
        - "explanation": str, plain-language text
        - "sources": list[str], the context snippets used
        - "sku": str
    """
    context = retrieve_context(sku, forecast, optimization)

    system_prompt, user_prompt = build_explanation_prompt(
        sku, forecast, optimization, context
    )

    explanation_text = _llm_client.call_llm(system_prompt, user_prompt)

    return {
        "sku": sku,
        "explanation": explanation_text,
        "sources": context,
    }
