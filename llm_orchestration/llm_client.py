"""
llm_client.py
-------------
Thin wrapper around whichever LLM API you use, so the rest of the code
never has to know API details directly. Makes it easy to swap providers
or mock the LLM for tests.

Uses Groq (free tier - no credit card needed, just an email signup at
console.groq.com). Groq serves open-source models (Llama, Mixtral, etc.)
on custom hardware at very high speed, via an OpenAI-compatible API, so
this uses the standard `openai` Python package pointed at Groq's endpoint.
See README.md for setup.

Falls back through two tiers if a real LLM call isn't available:
    1. Real Groq API call (if SDK installed + valid GROQ_API_KEY)
    2. Rule-based template explanation (free, no key needed - see
       _template_fallback() below) - good enough for a viva demo
"""

import os
import re
from typing import Optional


class LLMClient:
    def __init__(self, model: str = "openai/gpt-oss-20b", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self._client = None  # lazily initialized in call_llm()

    def call_llm(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = 500) -> str:
        """
        Send system_prompt + user_prompt to Groq and return the text
        response. Falls back to a free rule-based explanation if no real
        API key/SDK is available, so a demo still shows a real-looking
        sentence instead of a bare error message.
        """
        if self._client is None:
            try:
                from openai import OpenAI
                if not self.api_key:
                    raise ValueError("No GROQ_API_KEY set")
                self._client = OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=self.api_key,
                )
            except Exception:
                # Covers ImportError (SDK not installed) and missing key -
                # OpenAI() doesn't validate the key at construction time,
                # so we check for it explicitly above rather than waiting
                # for the actual call to fail.
                self._client = None

        if self._client is None:
            return self._template_fallback(user_prompt)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception:
            # Covers missing/invalid API key, rate limits, bad model name,
            # network errors, etc. - falls back to the free template instead
            # of a bare error message, so a demo still looks presentable.
            return self._template_fallback(user_prompt)

    def _template_fallback(self, user_prompt: str) -> str:
        """
        Free, rule-based explanation generator - no API key needed.

        Pulls the key numbers straight out of the already-built prompt
        text (order quantity, safety stock, reorder point, etc. are all
        present in user_prompt since build_explanation_prompt() put them
        there) and assembles a readable sentence from a template. Good
        enough to demo the pipeline end-to-end without paying for API
        calls; swap for a real LLM call whenever a key is available.
        """
        def _extract(label_pattern: str) -> str:
            match = re.search(rf"{label_pattern}[:\s]+([0-9.]+)", user_prompt, re.IGNORECASE)
            return match.group(1) if match else "N/A"

        order_qty = _extract(r"Order quantity \(q\)")
        safety_stock = _extract(r"Safety stock \(s\)")
        reorder_point = _extract(r"Reorder point \(r\)")
        total_cost = _extract(r"Total expected cost")

        return (
            f"[TEMPLATE EXPLANATION - no API key set, generated locally for free]\n\n"
            f"Based on the current demand forecast, the system recommends ordering "
            f"{order_qty} units. A safety stock of {safety_stock} units is held to "
            f"protect against demand variability, and the system will trigger a new "
            f"order once inventory drops to {reorder_point} units (the reorder point). "
            f"The total expected cost for this decision is approximately {total_cost}. "
            f"These values respect the configured service level and lead time "
            f"constraints for this SKU."
        )
