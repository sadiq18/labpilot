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

This module lives under ``accessor.common`` so both ``research_engine.intelligence`` and
``research_engine.execution`` can share one contract without ``execution``
importing ``intelligence`` (import-hygiene rule).
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from labpilot.accessor.common.provenance import record_invocation

if TYPE_CHECKING:
    from labpilot.llm.client import LLMClient

logger = logging.getLogger("labpilot.accessor.common.micro_agents")


#: What actually produced a result. Sourced from the branch taken in
#: :meth:`BaseMicroAgent.run`, never inferred by a caller.
GeneratedBy = Literal["llm", "rule_engine", "template_fallback", "stub"]

#: Opt-in escape hatch for deterministic operation. Without it, an agent with no
#: LLM refuses to run rather than silently substituting its rule engine — a
#: deterministic approximation presented as a result is how false findings enter
#: beliefs and claims.
DETERMINISTIC_ENV = "LABPILOT_DETERMINISTIC"

#: M14 phase 2b. When set, a *failed* LLM call raises instead of quietly
#: producing rule-engine output. Phase 2a already does this for a *missing*
#: client; 2b extends it to a client that answered badly.
#:
#: Default off, and the reason is measured rather than cautious. On rogii
#: 2026-08-07, a 27-step campaign recorded 28 invocations with a **11% fallback
#: rate, all three failures `json_shape`** — a model replying in prose, which is
#: exactly the failure 2b converts into a hard abort. At that rate a campaign
#: would die roughly every nine steps. Retrying with a corrective re-ask (see
#: `_is_transient_llm_error`) is the first move; flipping this default is the
#: second, once the recorded rate justifies it. `agent_invocations` holds the
#: number, so this is now a decision with evidence behind it rather than a
#: judgement call.
STRICT_LLM_ENV = "LABPILOT_STRICT_LLM"


class LLMDegradedError(RuntimeError):
    """The LLM was reachable and its answer was unusable (M14 2b).

    Distinct from :class:`LLMUnavailableError`, which means no client at all.
    Separate types because the operator responses differ: one is "configure a
    provider", the other is "this model cannot hold the contract".
    """


class LLMUnavailableError(RuntimeError):
    """No LLM configured, and deterministic mode was not requested.

    Raised rather than falling back because a missing LLM means *no reasoning
    happened*. Returning the rule engine's output would be indistinguishable
    from a reasoned result downstream.
    """


def deterministic_allowed() -> bool:
    """True when the operator explicitly asked for deterministic operation."""
    return os.environ.get(DETERMINISTIC_ENV, "").strip().lower() in {"1", "true", "yes"}


def strict_llm() -> bool:
    """True when a failed LLM call must raise rather than fall back (M14 2b)."""
    if os.environ.get(STRICT_LLM_ENV, "").strip().lower() in {"1", "true", "yes"}:
        # An explicit "be strict" outranks an explicit "deterministic is fine":
        # the operator asking for strictness is asking about *this* run.
        return True
    return False


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


def _complete_json(llm_client: object, system: str, user: str) -> str:
    """Ask for a completion, requesting constrained JSON decoding when supported.

    Every micro agent parses the reply as JSON, and small local models happily
    answer such prompts in prose instead — the reply is then discarded and the
    agent silently degrades to its rule engine. Providers that can enforce a
    JSON grammar should do so; those that cannot keep the old behaviour.
    """
    try:
        return llm_client.complete(system, user, json_mode=True)  # type: ignore[call-arg]
    except TypeError:
        return llm_client.complete(system, user)  # type: ignore[attr-defined]


#: The model answered, but not in the shape we asked for. Distinct from a rate
#: limit in the way that matters: waiting changes nothing, because nothing about
#: the provider is busy. Asking again — more firmly — sometimes does.
_SHAPE_MARKERS = (
    "DID NOT CONTAIN A JSON OBJECT",
    "JSONDECODEERROR",
    "EXPECTING VALUE",
    "VALIDATION ERROR",
)

_BUSY_MARKERS = (
    "429",
    "503",
    "UNAVAILABLE",
    "RESOURCE_EXHAUSTED",
    "RATE LIMIT",
    "HIGH DEMAND",
    "TEMPORARY",
    "TIMEOUT",
    "TIMED OUT",
)


def _is_shape_error(exc: BaseException) -> bool:
    return any(marker in str(exc).upper() for marker in _SHAPE_MARKERS)


def _is_transient_llm_error(exc: BaseException) -> bool:
    """True for failures worth another attempt.

    Shape failures were excluded, which mattered more than it looked: M14 phase
    2b makes LLM failure fatal, and ``Response did not contain a JSON object``
    is the failure actually observed in this system. Under 2b, with no retry, a
    single prose reply would abort a whole command. It is now retried — with a
    corrective instruction rather than the identical prompt, since repeating
    input that already produced prose is a poor bet.
    """
    text = str(exc).upper()
    return _is_shape_error(exc) or any(marker in text for marker in _BUSY_MARKERS)


def _retry_delay_for(exc: BaseException, configured: float) -> float:
    """Shape failures retry immediately; busy providers get the full backoff."""
    return 0.0 if _is_shape_error(exc) else configured


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
    #: What class of model this agent's prompt needs. Declared here rather than
    #: at the ~95 construction sites because the requirement belongs next to the
    #: prompt that creates it: whoever writes the prompt knows what it needs.
    llm_role: str = "reasoning"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        # Accept a gateway wherever a client is accepted, so every existing
        # `Agent(llm_client=...)` call site keeps working and plain test stubs
        # stay valid.
        if hasattr(llm_client, "for_role"):
            llm_client = llm_client.for_role(self.llm_role)  # type: ignore[union-attr]
        self.llm_client = llm_client
        self.last_used_llm = False
        self.last_generated_by: GeneratedBy = "rule_engine"
        self.last_failure_reason: str | None = None
        #: Set after a successful LLM call when the client reports it.
        self.last_served: object | None = None

    @property
    def uses_llm(self) -> bool:
        """Whether a client is *configured* — NOT whether the call succeeded.

        Do not use this for provenance. It reports True for a run that fell back
        to the rule engine, which is how an artifact came to record ``"llm"`` for
        deterministic output. Use :attr:`last_generated_by`.
        """
        return self.llm_client is not None

    def run(self, context: StructuredContext) -> BaseModel:
        if self.llm_client is None and not deterministic_allowed():
            raise LLMUnavailableError(
                f"{self.name or type(self).__name__} requires an LLM and none is "
                "configured. Check `research doctor`, start Ollama, or set "
                f"{DETERMINISTIC_ENV}=1 to accept deterministic rule-engine output."
            )
        self.last_used_llm = False
        self.last_served = None
        self._reask_reason: str | None = None
        # Default covers the silent path below: with no client configured the
        # try/except is skipped entirely and nothing is logged, so the only
        # record that this run was deterministic is the one set here.
        self.last_generated_by = "rule_engine"
        self.last_failure_reason = None if self.llm_client is not None else "no llm client"
        used_attempts = 0
        if self.llm_client is not None:
            max_attempts = max(1, int(getattr(self, "llm_max_attempts", 3)))
            retry_delay = float(getattr(self, "llm_retry_delay_seconds", 20.0))
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                used_attempts = attempt
                try:
                    result = self._run_llm(context)
                    self.last_used_llm = True
                    self.last_generated_by = "llm"
                    self.last_failure_reason = None
                    # Which model produced this, when the client can say. M14
                    # records *what kind of thing* ran; this records *which*.
                    self.last_served = getattr(self.llm_client, "last_served", None)
                    self._record_provenance(used_attempts)
                    return result
                except Exception as exc:  # noqa: BLE001 - soft-fail to deterministic path
                    last_exc = exc
                    if attempt < max_attempts and _is_transient_llm_error(exc):
                        delay = _retry_delay_for(exc, retry_delay)
                        if _is_shape_error(exc):
                            self._reask_reason = str(exc)
                        logger.warning(
                            "Micro agent %s LLM path failed (%s); retrying in %.0fs "
                            "(attempt %d/%d).",
                            self.name or type(self).__name__,
                            exc,
                            delay,
                            attempt,
                            max_attempts,
                        )
                        if delay > 0:
                            time.sleep(delay)
                        continue
                    logger.warning(
                        "Micro agent %s LLM path failed (%s); using rule_engine fallback.",
                        self.name or type(self).__name__,
                        last_exc,
                    )
                    self.last_failure_reason = str(last_exc)
                    if strict_llm():
                        self._record_provenance(attempt)
                        raise LLMDegradedError(
                            f"{self.name or type(self).__name__}: the LLM path failed "
                            f"after {attempt} attempt(s) and strict mode is on "
                            f"({STRICT_LLM_ENV}=1). Last failure: {last_exc}"
                        ) from last_exc
                    break
        # Recorded on the fallback path too — and on the silent no-client path,
        # which never enters the try/except above. A provenance record that only
        # exists when something interesting happened cannot produce a *rate*.
        self._record_provenance(used_attempts or 1)
        return self._run_rule_engine(context)

    def _record_provenance(self, attempts: int) -> None:
        record_invocation(
            agent=self.name or type(self).__name__,
            generated_by=self.last_generated_by,
            llm_role=str(getattr(self, "llm_role", "") or ""),
            failure_reason=self.last_failure_reason,
            attempts=attempts,
            served=self.last_served,
        )

    # --- LLM path ---------------------------------------------------------

    def system_prompt(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def user_prompt(self, context: StructuredContext) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def _run_llm(self, context: StructuredContext) -> BaseModel:
        assert self.llm_client is not None
        system = self.system_prompt()
        try:
            from pathlib import Path as _Path

            from labpilot.research_engine.shared.skills import compose_system_prompt

            agent_file = inspect.getfile(type(self))
            workspace = (
                context.data.get("workspace_root")
                or context.data.get("skill_overlay_dir")
                or ""
            )
            agent_key = str(context.data.get("skill_agent_key") or "").strip()
            if not agent_key:
                # Prefer package folder name (matches .labpilot/skills/<name>.md).
                agent_key = _Path(agent_file).resolve().parent.name
            if agent_key.endswith("Agent"):
                raw_key = agent_key[: -len("Agent")]
                agent_key = re.sub(r"(?<!^)(?=[A-Z])", "_", raw_key).lower()
            system = compose_system_prompt(
                system,
                agent_file=agent_file,
                workspace_root=workspace or None,
                agent_key=agent_key,
            )
        except Exception:  # noqa: BLE001 — skill injection must never break agents
            pass
        user = self.user_prompt(context)
        reason = getattr(self, "_reask_reason", None)
        if reason:
            # Repeating the identical prompt after a prose reply mostly produces
            # another prose reply. Naming the failure is what changes the odds.
            user = (
                f"{user}\n\n"
                "Your previous reply could not be parsed: "
                f"{reason[:200]}\n"
                "Reply with a single JSON object and nothing else — no prose, "
                "no explanation, no markdown fences."
            )
        raw = _complete_json(self.llm_client, system, user)
        return self._parse(raw)

    def _parse(self, raw: str) -> BaseModel:
        # Lazy import keeps ``common`` free of an import-time llm dependency.
        from labpilot.llm.json_utils import parse_json_object

        return self.output_model.model_validate(parse_json_object(raw))

    # --- deterministic path ----------------------------------------------

    def _run_rule_engine(self, context: StructuredContext) -> BaseModel:  # pragma: no cover
        raise NotImplementedError
