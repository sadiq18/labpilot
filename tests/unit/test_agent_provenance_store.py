"""Durable micro-agent provenance — the data M14 2b and 3 are blocked on.

Companion to `test_agent_provenance.py`, which covers phase 1's in-memory
stamps. This file covers making them survive the run.

Phase 1 made each agent report what produced its output; the report lived on the
instance and died with it. The only durable trace was
``research_plans.generated_by``, covering 1 of the 21 agents implementing
``_run_rule_engine``.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from labpilot.accessor.common.micro_agents import (
    DETERMINISTIC_ENV,
    BaseMicroAgent,
    StructuredContext,
)
from labpilot.accessor.common.provenance import (
    AgentInvocation,
    classify_failure,
    set_run_context,
    set_sink,
)
from labpilot.research_engine.telemetry.agent_provenance import (
    invocation_totals,
    llm_failure_report,
    recording_provenance,
    rule_engine_fire_report,
)

COMPETITION = "prov-demo"


class _Out(BaseModel):
    value: str = "x"


class _Agent(BaseMicroAgent):
    name = "demo_agent"
    llm_role = "reasoning"

    def __init__(self, client=None, *, fail: Exception | None = None) -> None:
        super().__init__(llm_client=client)
        self._fail = fail
        self.llm_max_attempts = 1

    def system_prompt(self) -> str:
        return "s"

    def user_prompt(self, context) -> str:
        return "u"

    def _run_llm(self, context):
        if self._fail:
            raise self._fail
        return _Out(value="llm")

    def _run_rule_engine(self, context):
        return _Out(value="rules")


class _Client:
    last_served = None

    def complete(self, *a, **k):
        return "{}"


class _Recorder:
    def __init__(self) -> None:
        self.records: list[AgentInvocation] = []

    def record(self, invocation: AgentInvocation) -> None:
        self.records.append(invocation)


@pytest.fixture
def recorder():
    rec = _Recorder()
    set_sink(rec)
    set_run_context(competition=COMPETITION, session_id="S-1")
    yield rec
    set_sink(None)
    set_run_context()


# --- the failure taxonomy ---------------------------------------------------


@pytest.mark.parametrize(
    ("reason", "kind"),
    [
        ("Response did not contain a JSON object. Got: 'Sure! Here...'", "json_shape"),
        ("json.decoder.JSONDecodeError: Expecting value", "json_shape"),
        ("429 Too Many Requests", "rate_limit"),
        ("RESOURCE_EXHAUSTED", "rate_limit"),
        ("503 Service Unavailable", "unavailable"),
        ("Read timed out", "timeout"),
        ("401 Unauthorized", "auth"),
        ("no llm client", "no_client"),
        ("something nobody has seen", "other"),
        (None, None),
        ("", None),
    ],
)
def test_failures_are_bucketed(reason, kind):
    assert classify_failure(reason) == kind


def test_json_shape_wins_over_broader_buckets():
    """A prose reply quoting a 429 in its text is still a shape failure."""
    reason = "Response did not contain a JSON object. Got: 'the API returned 429'"
    assert classify_failure(reason) == "json_shape"


# --- recording every path ---------------------------------------------------


def test_a_successful_llm_run_is_recorded(recorder):
    _Agent(_Client()).run(StructuredContext(data={}))
    (rec,) = recorder.records
    assert rec.agent == "demo_agent"
    assert rec.generated_by == "llm"
    assert rec.failure_kind is None
    assert rec.llm_role == "reasoning"
    assert rec.competition_slug == COMPETITION
    assert rec.session_id == "S-1"


def test_a_fallback_is_recorded_with_its_reason(recorder):
    _Agent(_Client(), fail=ValueError("Response did not contain a JSON object")).run(
        StructuredContext(data={})
    )
    (rec,) = recorder.records
    assert rec.generated_by == "rule_engine"
    assert rec.failure_kind == "json_shape"


def test_the_silent_no_client_path_is_recorded(recorder, monkeypatch):
    """This path never enters the try/except, so it recorded nothing at all.

    A provenance record that only appears when something interesting happened
    cannot produce a *rate* — the denominator has to be written too.
    """
    monkeypatch.setenv(DETERMINISTIC_ENV, "1")
    _Agent(None).run(StructuredContext(data={}))
    (rec,) = recorder.records
    assert rec.generated_by == "rule_engine"
    assert rec.failure_kind == "no_client"


def test_no_sink_installed_is_a_no_op():
    """Telemetry must never be able to break a run that would have worked."""
    set_sink(None)
    assert _Agent(_Client()).run(StructuredContext(data={})).value == "llm"


def test_a_broken_sink_does_not_break_the_agent():
    class _Boom:
        def record(self, invocation):
            raise RuntimeError("disk full")

    set_sink(_Boom())
    try:
        assert _Agent(_Client()).run(StructuredContext(data={})).value == "llm"
    finally:
        set_sink(None)


# --- the reports 2b and 3 consume -------------------------------------------


def test_reports_over_a_recorded_campaign(tmp_path):
    with recording_provenance(tmp_path, COMPETITION, session_id="S-1"):
        _Agent(_Client()).run(StructuredContext(data={}))
        _Agent(_Client()).run(StructuredContext(data={}))
        _Agent(_Client(), fail=ValueError("did not contain a JSON object")).run(
            StructuredContext(data={})
        )
        _Agent(_Client(), fail=RuntimeError("429 rate limit")).run(StructuredContext(data={}))

    totals = invocation_totals(tmp_path, COMPETITION)
    assert totals == {"total": 4, "llm": 2, "rule_engine": 2}

    failures = llm_failure_report(tmp_path, COMPETITION)
    assert failures == {"json_shape": 1, "rate_limit": 1}

    (stat,) = rule_engine_fire_report(tmp_path, COMPETITION)
    assert stat.agent == "demo_agent"
    assert stat.total == 4
    assert stat.rule_engine == 2
    assert stat.fallback_rate == pytest.approx(0.5)


def test_reports_are_empty_without_a_campaign(tmp_path):
    assert rule_engine_fire_report(tmp_path, COMPETITION) == []
    assert llm_failure_report(tmp_path, COMPETITION) == {}
    assert invocation_totals(tmp_path, COMPETITION) == {"total": 0, "llm": 0, "rule_engine": 0}


def test_the_sink_is_uninstalled_afterwards(tmp_path):
    with recording_provenance(tmp_path, COMPETITION):
        pass
    _Agent(_Client()).run(StructuredContext(data={}))
    assert invocation_totals(tmp_path, COMPETITION)["total"] == 0


# --- shape failures: retried, not waited on (M14 2b de-risking) -------------


def test_a_shape_failure_is_retryable():
    """`Response did not contain a JSON object` is the failure this system
    actually produces, and it got no retry — so under 2b one prose reply would
    abort a whole command."""
    from labpilot.accessor.common.micro_agents import (
        _is_shape_error,
        _is_transient_llm_error,
    )

    exc = ValueError("Response did not contain a JSON object. Got: 'Sure!'")
    assert _is_shape_error(exc)
    assert _is_transient_llm_error(exc)


def test_a_shape_failure_does_not_wait():
    """Nothing is busy, so backing off buys nothing but lost campaign time."""
    from labpilot.accessor.common.micro_agents import _retry_delay_for

    assert _retry_delay_for(ValueError("did not contain a JSON object"), 20.0) == 0.0
    assert _retry_delay_for(RuntimeError("429 rate limit"), 20.0) == 20.0


def test_a_fatal_error_is_still_not_retried():
    from labpilot.accessor.common.micro_agents import _is_transient_llm_error

    assert not _is_transient_llm_error(RuntimeError("401 Unauthorized"))
    assert not _is_transient_llm_error(RuntimeError("model exploded"))
