"""
rag.py
------
Retrieval-Augmented Generation logic.

Responsibility: given the current SKU/context, retrieve relevant supporting
information (historical demand patterns, business rules, past decisions,
policy documents, etc.) that the LLM can use to ground its explanation.

TODO (Harshitha):
1. Decide what your "knowledge base" actually is. Options, roughly in
   order of implementation effort:
   a. A handful of markdown/text files with business rules
      (e.g. "safety stock should never exceed 30% of average monthly demand")
   b. Historical SKU-level stats pulled straight from the same CSV
      Sam and Bhuvii already use (retail_store_inventory.csv)
   c. A proper vector store (embeddings + similarity search) - see
      vector_store.py if you go this route. This is the most impressive
      for a viva but also the most work; only worth it if time permits.
2. Implement retrieve_context() below to return a list of short text
   snippets relevant to the current SKU / situation.
"""

from typing import List
from shared.schemas import ForecastOutput, OptimizationOutput


def retrieve_context(sku: str, forecast: ForecastOutput,
                      optimization: OptimizationOutput,
                      top_k: int = 3) -> List[str]:
    """
    Retrieve up to `top_k` relevant context snippets for this SKU/decision.

    Parameters
    ----------
    sku : str
        e.g. "S001::P0001"
    forecast : ForecastOutput
        Sam's forecast_demand() output, wrapped in the shared schema.
    optimization : OptimizationOutput
        Bhuvii's optimize_inventory() output.
    top_k : int
        Max number of context snippets to return.

    Returns
    -------
    List[str]
        Plain text snippets to insert into the LLM prompt.
    """
    # PLACEHOLDER: replace with real retrieval logic.
    # Simple starting point - static business-rule snippets.
    context = [
        "Business rule: maintain at least a 90% service level for all SKUs.",
        "Business rule: safety stock should scale with demand variability, "
        "not just lead time.",
        f"Historical note: SKU {sku} has been analyzed previously; "
        f"compare current recommendation against past trend if available.",
    ]
    return context[:top_k]
