"""Durable record of what actually produced each micro-agent result.

M14 phase 1 made every agent *report* whether the LLM or its rule engine produced
the output. The report lived on the agent instance and died with it: the only
trace that outlived a run was ``research_plans.generated_by``, covering one of the
21 agents that implement ``_run_rule_engine``. Measured on rogii, that one column
already says something worth knowing — 10 of 19 plans came from the rule engine —
and there was no way to ask the same question of the other twenty.

Both remaining M14 phases are blocked on exactly that gap:

* **2b** makes LLM failure fatal, and needs the rate at which the LLM path fails
  *and with which error* before that is safe. The failure actually observed —
  ``Response did not contain a JSON object`` — is not a rate-limit error and gets
  no retry, so under 2b one prose reply would abort a command.
* **3** triages the rule engines, and needs them ranked by fire rate: one that
  never fires is dead code, one that fires constantly is either load-bearing
  domain logic or masking a persistent LLM failure.

Neither question is answerable from logs afterwards, so the record is written as
it happens.

**Layering.** `BaseMicroAgent` lives in ``accessor/`` and must not import the
research engine's stores. So this module defines the shape of a record and a sink
to receive it; the research engine installs a sink that writes to SQLite. With no
sink installed — unit tests, library use — recording is a no-op, which is why
this cannot break a run that would otherwise have worked.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentInvocation:
    """One micro-agent run, and what produced its result."""

    agent: str
    generated_by: str
    llm_role: str = ""
    failure_reason: str | None = None
    failure_kind: str | None = None
    attempts: int = 1
    provider: str | None = None
    model: str | None = None
    latency_ms: int | None = None
    competition_slug: str = ""
    session_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class InvocationSink(Protocol):
    def record(self, invocation: AgentInvocation) -> None: ...


_sink: ContextVar[InvocationSink | None] = ContextVar("_provenance_sink", default=None)
_competition: ContextVar[str] = ContextVar("_provenance_competition", default="")
_session: ContextVar[str | None] = ContextVar("_provenance_session", default=None)


def set_sink(sink: InvocationSink | None) -> Token:
    """Install ``sink`` and return a token that restores the previous one."""
    return _sink.set(sink)


def reset_sink(token: Token) -> None:
    """Put back whatever sink was installed before the matching `set_sink`.

    Clearing to ``None`` instead would drop the process-wide sink that
    `cli/main.py` installs, so any agent work continuing after a campaign
    would stop being recorded.
    """
    _sink.reset(token)


def set_run_context(*, competition: str = "", session_id: str | None = None) -> tuple:
    """Tag subsequent records with the campaign they belong to."""
    return _competition.set(competition or ""), _session.set(session_id)


def reset_run_context(tokens: tuple) -> None:
    _competition.reset(tokens[0])
    _session.reset(tokens[1])


#: Failure taxonomy. The point is to separate "the provider was busy" from "the
#: model answered, but not in the shape we asked for" — they call for opposite
#: responses, and M14 2b hinges on the second one's rate.
_KINDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Checked first: a JSON-shape failure often mentions a model or provider
    # name that would otherwise match a broader bucket.
    ("json_shape", ("DID NOT CONTAIN A JSON OBJECT", "JSONDECODEERROR", "EXPECTING VALUE")),
    ("schema", ("VALIDATIONERROR", "FIELD REQUIRED", "PYDANTIC")),
    ("rate_limit", ("429", "RATE LIMIT", "RESOURCE_EXHAUSTED", "HIGH DEMAND", "QUOTA")),
    ("unavailable", ("503", "UNAVAILABLE", "TEMPORARY", "502", "500")),
    ("timeout", ("TIMEOUT", "TIMED OUT")),
    ("auth", ("401", "403", "UNAUTHORIZED", "FORBIDDEN", "API KEY")),
    ("no_client", ("NO LLM CLIENT",)),
)


def classify_failure(reason: str | None) -> str | None:
    """Bucket a failure message, or None when there was no failure.

    Deliberately coarse and ordered: this feeds a rate, not a diagnosis. An
    unrecognised message is ``other`` rather than absent, so a growing ``other``
    count is itself a signal that the taxonomy has fallen behind.
    """
    if not reason:
        return None
    text = str(reason).upper()
    for kind, markers in _KINDS:
        if any(marker in text for marker in markers):
            return kind
    return "other"


def record_invocation(
    *,
    agent: str,
    generated_by: str,
    llm_role: str = "",
    failure_reason: str | None = None,
    attempts: int = 1,
    served: object | None = None,
) -> None:
    """Write one record, if a sink is installed. Never raises."""
    sink = _sink.get()
    if sink is None:
        return
    try:
        sink.record(
            AgentInvocation(
                agent=agent,
                generated_by=generated_by,
                llm_role=llm_role,
                failure_reason=failure_reason,
                failure_kind=classify_failure(failure_reason),
                attempts=attempts,
                provider=getattr(served, "provider", None),
                model=getattr(served, "model", None),
                latency_ms=getattr(served, "latency_ms", None),
                competition_slug=_competition.get(),
                session_id=_session.get(),
            )
        )
    except Exception as exc:  # noqa: BLE001 — telemetry must never break a run
        logger.debug("provenance record dropped: %s", exc)
