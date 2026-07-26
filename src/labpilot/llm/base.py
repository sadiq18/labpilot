"""Abstract LLM provider interface.

Everything outside ``labpilot.llm`` talks to :class:`~labpilot.llm.client.LLM`
(or the legacy :class:`~labpilot.llm.client.LLMClient` protocol). Providers
implement this low-level completion contract only.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Provider(Protocol):
    """One-shot system+user completion for a concrete backend."""

    name: str

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float,
    ) -> str: ...
