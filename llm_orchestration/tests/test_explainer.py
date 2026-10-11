"""
test_explainer.py
------------------
Basic tests for the explanation pipeline. Run with:
    pytest tests/test_explainer.py -v

These tests avoid requiring a live LLM API key by relying on the stub
fallback in llm_client.py - they check structure/format, not the exact
wording of the LLM's response.
"""

import pytest
from llm_orchestration.main import generate_explanation
from llm_orchestration.rag import retrieve_context
from llm_orchestration.prompt_templates import build_explanation_prompt
from shared.schemas import ForecastOutput, OptimizationOutput


SAMPLE_FORECAST = ForecastOutput(
    sku_id="S001::P0001", horizon_days=14,
    point_forecast=[45.2] * 14, lower_bound=[36.0] * 14,
    upper_bound=[54.0] * 14, model_confidence=0.8,
)
SAMPLE_OPTIMIZATION = OptimizationOutput(
    order_quantity=320.0, safety_stock=85.0, reorder_point=210.0,
    total_expected_cost=1450.75, solver_status="optimal",
)


def test_retrieve_context_returns_list_of_strings():
    context = retrieve_context("S001::P0001", SAMPLE_FORECAST, SAMPLE_OPTIMIZATION)
    assert isinstance(context, list)
    assert all(isinstance(c, str) for c in context)
    assert len(context) > 0


def test_build_prompt_contains_key_numbers():
    context = retrieve_context("S001::P0001", SAMPLE_FORECAST, SAMPLE_OPTIMIZATION)
    system_prompt, user_prompt = build_explanation_prompt(
        "S001::P0001", SAMPLE_FORECAST, SAMPLE_OPTIMIZATION, context
    )
    assert "320.00" in user_prompt  # order_quantity
    assert "85.00" in user_prompt   # safety_stock
    assert "210.00" in user_prompt  # reorder_point
    assert isinstance(system_prompt, str) and len(system_prompt) > 0


def test_generate_explanation_returns_expected_keys():
    result = generate_explanation("S001::P0001", SAMPLE_FORECAST, SAMPLE_OPTIMIZATION)
    assert set(result.keys()) == {"sku", "explanation", "sources"}
    assert result["sku"] == "S001::P0001"
    assert isinstance(result["explanation"], str)
    assert len(result["explanation"]) > 0
    assert isinstance(result["sources"], list)


def test_forecast_output_requires_fields():
    # Now that we use the real shared.schemas objects, Pydantic enforces
    # required fields at construction time - this is a feature, not a bug:
    # a malformed forecast fails fast and loudly instead of silently
    # producing a garbage explanation downstream.
    with pytest.raises(Exception):
        ForecastOutput(sku_id="S002::P0002")  # missing required fields
