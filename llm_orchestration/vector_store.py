"""
vector_store.py
----------------
OPTIONAL - only needed if you want embeddings-based similarity search
instead of the simple static/rule-based retrieval in rag.py.

If your build spec / viva scope doesn't require real embeddings, you can
leave this file mostly empty and just use rag.py's static approach.
Keeping this file as a stub shows the panel you considered the design
choice deliberately.

TODO (Harshitha, optional):
1. Pick an embedding source (e.g. sentence-transformers locally, or an
   API-based embedding model).
2. Build an index (even a simple in-memory list of (text, vector) pairs
   is fine for a project of this size - no need for a full vector DB).
3. Implement add_documents() and search() below.
"""

from typing import List, Tuple


class SimpleVectorStore:
    """A minimal in-memory vector store (no external DB required)."""

    def __init__(self):
        self._docs: List[str] = []
        self._vectors: List[List[float]] = []

    def add_documents(self, docs: List[str]) -> None:
        """Embed and store a list of text documents."""
        # PLACEHOLDER: compute real embeddings here.
        raise NotImplementedError("Embed docs and append to self._docs / self._vectors")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Return top_k (document, similarity_score) pairs for the query."""
        # PLACEHOLDER: embed query, compute cosine similarity, sort, return top_k.
        raise NotImplementedError("Implement similarity search")
