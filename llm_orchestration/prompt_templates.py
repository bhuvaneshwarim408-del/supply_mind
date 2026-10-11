"""
prompt_templates.py
--------------------
Central place for all prompt strings sent to the LLM.
Keeping these separate from explainer.py makes it easy to tweak wording
without touching the orchestration logic - and easy to show the panel
exactly how you engineered the prompt.
"""

EXPLANATION_SYSTEM_PROMPT = """\
You are an inventory management assistant. You will be given:
1. A demand forecast for a product (SKU).
2. An optimized inventory decision (order quantity, safety stock, reorder point).
3. Supporting context (business rules and/or historical notes).

Your job: explain the decision in plain, confident business language that a
non-technical store manager could understand. Be concise (3-5 sentences).
Do not invent numbers that were not provided. Reference the forecast trend
and the reasoning behind the safety stock / reorder point if relevant context
is given.
"""

EXPLANATION_USER_TEMPLATE = """\
SKU: {sku}

Forecast:
- Horizon: {horizon} days
- Predicted average demand: {avg_demand:.2f} units/day
- Trend: {trend}

Optimization result:
- Order quantity (q): {order_quantity:.2f}
- Safety stock (s): {safety_stock:.2f}
- Reorder point (r): {reorder_point:.2f}
- Solver status: {solver_status}
- Total expected cost: {total_cost:.2f}

Relevant context:
{context_block}

Write a short, clear explanation of this inventory recommendation.
"""


def build_explanation_prompt(sku: str, forecast: "ForecastOutput",
                              optimization: "OptimizationOutput",
                              context: list) -> tuple:
    """
    Build (system_prompt, user_prompt) strings ready to send to the LLM.

    Takes the team's real shared.schemas objects directly - no adapter
    needed on Ruchitha's side anymore.
    """
    context_block = "\n".join(f"- {c}" for c in context) if context else "- (none)"

    avg_demand = sum(forecast.point_forecast) / len(forecast.point_forecast)
    trend = _infer_trend(forecast.point_forecast)

    user_prompt = EXPLANATION_USER_TEMPLATE.format(
        sku=sku,
        horizon=forecast.horizon_days,
        avg_demand=avg_demand,
        trend=trend,
        order_quantity=optimization.order_quantity,
        safety_stock=optimization.safety_stock,
        reorder_point=optimization.reorder_point,
        solver_status=optimization.solver_status,
        total_cost=optimization.total_expected_cost,
        context_block=context_block,
    )
    return EXPLANATION_SYSTEM_PROMPT, user_prompt


def _infer_trend(point_forecast: list) -> str:
    """
    shared.schemas.ForecastOutput has no trend field, so this derives a
    simple rising/falling/stable estimate from the forecast curve itself.
    Flag as a known placeholder if asked in viva - a real trend signal
    would ideally come from the forecasting model.
    """
    if not point_forecast or len(point_forecast) < 2:
        return "stable"
    mid = len(point_forecast) // 2
    first_half_avg = sum(point_forecast[:mid]) / mid
    second_half_avg = sum(point_forecast[mid:]) / (len(point_forecast) - mid)
    if second_half_avg > first_half_avg * 1.05:
        return "rising"
    if second_half_avg < first_half_avg * 0.95:
        return "falling"
    return "stable"
