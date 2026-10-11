"""Generate an answer from retrieved context with Claude."""

from __future__ import annotations

from .llm import call_text, llm_available
from .retrieve import Retrieval

SYSTEM_PROMPT = """You answer questions about an organisation's product development processes.
Use only the provided context, which was retrieved from a process knowledge graph or from the process documents.
Name the concrete activities, roles, artifacts and tools involved, and mention the source document where one is given.
If the context does not contain the answer, say what is missing instead of guessing."""


def answer(retrieval: Retrieval) -> str | None:
    """Return the generated answer, or None when no LLM credentials are configured."""
    if not llm_available():
        return None
    user = f"Context:\n{retrieval.context}\n\nQuestion: {retrieval.question}"
    return call_text(SYSTEM_PROMPT, user)
