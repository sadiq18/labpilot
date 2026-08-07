"""Durable micro-agent provenance — the data M14 2b and 3 are blocked on.

Companion to `test_agent_provenance.py`, which covers in-memory stamps. This
file covers making them survive the run.

Issue #39 removed rule-engine substitutes: a failed LLM call raises
``LLMDegradedError`` after recording the attempt; a missing client raises
``LLMUnavailableError`` without producing an artifact.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from labpilot.accessor.common.micro_agents import (
    BaseMicroAgent,
    LLMDegradedError,
    LLMUnavailableError,
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
    output_model = _Out
    llm_role = "reasoning"

    def __init__(self, client=None, *, fail: Exception | None = None) -> None:
        super().__init__(llm_client=client)
        self._fail = fail
        self.llm_max_attempts = 1
        self.llm_retry_delay_seconds = 0.0

    def system_prompt(self) -> str:
        return "s"

    def user_prompt(self, context) -> str:
        return "u"

    def _run_llm(self, context):
        if self._fail:
            raise self._fail
        return _Out(value="llm")


class _Client:
    last_served = None

    def complete(self, *a, **k):
        return '{"value": "llm"}'


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


def test_a_degraded_call_is_recorded_with_its_reason(recorder):
    with pytest.raises(LLMDegradedError):
        _Agent(_Client(), fail=ValueError("Response did not contain a JSON object")).run(
            StructuredContext(data={})
        )
    (rec,) = recorder.records
    assert rec.generated_by == "llm"
    assert rec.failure_kind == "json_shape"


def test_no_client_raises_without_an_artifact(recorder):
    """Missing client is a hard refuse — no invented output, no success stamp."""
    with pytest.raises(LLMUnavailableError):
        _Agent(None).run(StructuredContext(data={}))
    assert recorder.records == []


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
        with pytest.raises(LLMDegradedError):
            _Agent(_Client(), fail=ValueError("did not contain a JSON object")).run(
                StructuredContext(data={})
            )
        with pytest.raises(LLMDegradedError):
            _Agent(_Client(), fail=RuntimeError("429 rate limit")).run(
                StructuredContext(data={})
            )

    totals = invocation_totals(tmp_path, COMPETITION)
    # Failures still stamp generated_by=llm (attempted); rule_engine fires are gone.
    assert totals == {"total": 4, "llm": 4, "rule_engine": 0}

    failures = llm_failure_report(tmp_path, COMPETITION)
    assert failures == {"json_shape": 1, "rate_limit": 1}

    assert all(s.rule_engine == 0 for s in rule_engine_fire_report(tmp_path, COMPETITION))


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


def test_the_conductor_policy_records_its_own_failures(tmp_path):
    """The policy is the highest-frequency LLM caller and is not a micro agent.

    Measured on rogii 2026-08-07: an 8-step campaign recorded one invocation
    while the policy fell back to the offline order three times — so the very
    number M14 2b needs was the number missing.
    """
    import labpilot.research_engine.conductor.policy as policy_mod

    class _Failing:
        def complete(self, system, user):
            raise RuntimeError("returned no choices")

    with recording_provenance(tmp_path, COMPETITION, session_id="S-9"):
        with pytest.raises(Exception):
            policy_mod._invoke_llm_next_action(  # noqa: SLF001
                {"allowlist": ["stop"]}, {"stop"}, _Failing()
            )

    report = rule_engine_fire_report(tmp_path, COMPETITION)
    assert any(s.agent == "ConductorPolicy" for s in report)


# --- review #96: the campaign sink must not clobber the CLI sink ------------


def test_the_campaign_sink_restores_the_cli_sink(tmp_path):
    """`cli/main.py` installs a process-wide sink; clearing to None on campaign
    exit would silently stop recording anything that runs afterwards."""
    outer = _Recorder()
    token = set_sink(outer)
    try:
        with recording_provenance(tmp_path, COMPETITION, session_id="S-1"):
            _Agent(_Client()).run(StructuredContext(data={}))
        # Back to the outer sink, not to None.
        _Agent(_Client()).run(StructuredContext(data={}))
        assert len(outer.records) == 1
    finally:
        set_sink(None)
        _ = token


def test_run_context_is_restored_too(tmp_path):
    from labpilot.accessor.common.provenance import set_run_context

    outer = _Recorder()
    set_sink(outer)
    set_run_context(competition="outer-comp", session_id="OUTER")
    try:
        with recording_provenance(tmp_path, COMPETITION, session_id="S-1"):
            pass
        _Agent(_Client()).run(StructuredContext(data={}))
        assert outer.records[-1].competition_slug == "outer-comp"
        assert outer.records[-1].session_id == "OUTER"
    finally:
        set_sink(None)
        set_run_context()
