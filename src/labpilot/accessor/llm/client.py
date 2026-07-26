"""Accessor entry point for the LLM client.

Re-exports the provider-agnostic LLM client so pillars (planner / intelligence /
execution) depend on ``labpilot.accessor.llm`` rather than importing each other.
The implementation stays in :mod:`labpilot.llm.client` (its module-level names
are patched by existing tests); this module is a thin, stable facade.
"""

from labpilot.llm.client import (
    LLMClient,
    complete_with_fallback,
    create_llm_client,
    llm_setup_hints,
    resolve_llm_client,
)

__all__ = [
    "LLMClient",
    "complete_with_fallback",
    "create_llm_client",
    "llm_setup_hints",
    "resolve_llm_client",
]
