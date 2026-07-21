"""Micro Agent core contract (design §2.4 "Micro Agents (locked)").

A Micro Agent is a *tiny specialized reasoning function* inside the Reasoning
Engine::

    input → prompt → typed artifact (structured output)

It is **not** an autonomous agent: no memory, no planning, no loops, no
network, no side effects. The caller persists whatever it returns.

Selective-LLM policy: Micro Agents are an **optional upgrade**. Every agent
must produce a valid typed artifact with no LLM configured by falling back to a
deterministic ``rule_engine`` path (same posture as the brief / reflection
templates). Passing ``llm_client=None`` disables the LLM and is the default in
tests and CI.

This module lives under ``common`` so both ``research_engine.intelligence`` and
``research_engine.execution`` can share one contract without ``execution``
importing ``intelligence`` (import-hygiene rule).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from labpilot.llm.client import LLMClient

logger = logging.getLogger("labpilot.common.micro_agents")


def coerce_str_list(value: object) -> list[str]:
    """Normalize a loose value into a clean ``list[str]``.

    Used by ``rule_engine`` paths that read pre-parsed signals from
    :attr:`StructuredContext.data`, which may be missing, a scalar, or a list.
    """
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


class StructuredContext(BaseModel):
    """Typed envelope the Deterministic Engine hands to a Micro Agent.

    The LLM never sees the raw database, Kaggle, GitHub, or arXiv — only this
    already-retrieved, already-compressed context (design §2.4 hard rule).
    Fields are intentionally generic so every agent shares one input type;
    each agent reads the subset it needs.
    """

    competition: str = ""
    question: str = ""
    # Free-form source text to extract from (paper body, repo readme, forum
    # thread). Deterministically fetched + normalized upstream.
    text: str = ""
    # Structured signals already computed by the Deterministic Engine
    # (metrics, pipeline diff, pre-parsed lists, …). The ``rule_engine`` path
    # reads from here so it stays deterministic and offline.
    data: dict[str, Any] = Field(default_factory=dict)
    # Candidate strings for set-shaped tasks (e.g. concept normalization).
    items: list[str] = Field(default_factory=list)


@runtime_checkable
class MicroAgent(Protocol):
    """One reasoning slice: typed context in, typed Pydantic artifact out."""

    name: str

    def run(self, context: StructuredContext) -> BaseModel:
        """prompt → LLM | rule_engine → validated typed artifact."""
        ...


class BaseMicroAgent:
    """Convenience base implementing the LLM → ``rule_engine`` fallback.

    Subclasses set :attr:`name` and :attr:`output_model`, implement
    :meth:`system_prompt` / :meth:`user_prompt` for the LLM path, and
    :meth:`_run_rule_engine` for the deterministic path. If no ``llm_client``
    is configured (or the LLM path raises / returns invalid JSON), the agent
    quietly downgrades to ``rule_engine`` — never a hard error.
    """

    name: str = ""
    #: The Pydantic type this agent always returns.
    output_model: type[BaseModel]

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    @property
    def uses_llm(self) -> bool:
        return self.llm_client is not None

    def run(self, context: StructuredContext) -> BaseModel:
        if self.llm_client is not None:
            try:
                return self._run_llm(context)
            except Exception as exc:  # noqa: BLE001 - soft-fail to deterministic path
                logger.warning(
                    "Micro agent %s LLM path failed (%s); using rule_engine fallback.",
                    self.name or type(self).__name__,
                    exc,
                )
        return self._run_rule_engine(context)

    # --- LLM path ---------------------------------------------------------

    def system_prompt(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def user_prompt(self, context: StructuredContext) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def _run_llm(self, context: StructuredContext) -> BaseModel:
        assert self.llm_client is not None
        raw = self.llm_client.complete(self.system_prompt(), self.user_prompt(context))
        return self._parse(raw)

    def _parse(self, raw: str) -> BaseModel:
        # Lazy import keeps ``common`` free of an import-time llm dependency.
        from labpilot.llm.json_utils import parse_json_object

        return self.output_model.model_validate(parse_json_object(raw))

    # --- deterministic path ----------------------------------------------

    def _run_rule_engine(self, context: StructuredContext) -> BaseModel:  # pragma: no cover
        raise NotImplementedError
