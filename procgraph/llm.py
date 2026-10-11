"""Thin wrapper around the Claude API.

The LLM is optional: everything except `extract --llm` and generated answers
works offline. Credentials are resolved by the Anthropic SDK (for example the
ANTHROPIC_API_KEY environment variable).
"""

from __future__ import annotations

import os

MODEL = os.environ.get("PROCGRAPH_MODEL", "claude-opus-5-5")

# If the model declines a request, let the API retry it on a suitable fallback model.
FALLBACK_KWARGS = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}


def llm_available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _client():
    import anthropic

    return anthropic.Anthropic()


def _check_refusal(response) -> None:
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        raise RuntimeError(f"the model declined the request: {getattr(details, 'explanation', '')}")


def call_parse(system: str, user: str, output_format):
    """Structured output: returns an instance of the given Pydantic model."""
    response = _client().beta.messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=output_format,
        output_config={"effort": "medium"},
        **FALLBACK_KWARGS,
    )
    _check_refusal(response)
    return response.parsed_output


def call_text(system: str, user: str) -> str:
    response = _client().beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": "medium"},
        **FALLBACK_KWARGS,
    )
    _check_refusal(response)
    return "".join(block.text for block in response.content if block.type == "text")
