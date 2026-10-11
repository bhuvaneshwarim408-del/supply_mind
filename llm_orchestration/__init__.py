"""
LLM Orchestration & RAG Explanation Module
Member 3: Harshitha N

Takes forecast output (from Sam's module) and optimization output
(from Bhuvii's module) and produces a plain-language, retrieval-grounded
explanation of the inventory decision.
"""

from .main import generate_explanation

__all__ = ["generate_explanation"]
