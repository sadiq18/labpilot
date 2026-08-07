"""M14 — every result records what actually produced it; failures raise.

Issue #39 removed rule-engine substitutes. Missing or failing clients raise
``LLMUnavailableError`` / ``LLMDegradedError`` instead of inventing output.
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


class _Out(BaseModel):
    value: str = "x"


class _Agent(BaseMicroAgent):
    name = "prov_probe"
    output_model = _Out
    llm_max_attempts = 1
    llm_retry_delay_seconds = 0.0

    def system_prompt(self) -> str:
        return "s"

    def user_prompt(self, context: StructuredContext) -> str:
        return "u"


class _GoodClient:
    def complete(self, system, user, *, json_mode=False):
        return '{"value": "llm"}'


class _BadClient:
    def complete(self, system, user, *, json_mode=False):
        return "These rules outline the guidelines for participating..."


def _run(agent):
    return agent.run(StructuredContext())


def test_llm_success_records_llm():
    agent = _Agent(llm_client=_GoodClient())
    result = _run(agent)
    assert result.value == "llm"
    assert agent.last_generated_by == "llm"
    assert agent.last_failure_reason is None


def test_llm_failure_raises_degraded():
    """A model answering a JSON prompt in prose aborts — no substitute."""
    agent = _Agent(llm_client=_BadClient())
    with pytest.raises(LLMDegradedError) as exc:
        _run(agent)
    assert "prov_probe" in str(exc.value)
    assert agent.last_failure_reason
    assert "JSON" in agent.last_failure_reason


def test_no_client_raises_unavailable():
    agent = _Agent(llm_client=None)
    with pytest.raises(LLMUnavailableError) as exc:
        _run(agent)
    message = str(exc.value)
    assert "prov_probe" in message
    assert "research doctor" in message


def test_provenance_resets_between_runs():
    agent = _Agent(llm_client=_GoodClient())
    _run(agent)
    assert agent.last_generated_by == "llm"
    assert agent.last_failure_reason is None

    agent.llm_client = _BadClient()
    with pytest.raises(LLMDegradedError):
        _run(agent)
    assert agent.last_failure_reason, "stale success must not clear the failure"


def test_uses_llm_is_not_provenance():
    """`uses_llm` is "a client exists", not "the call succeeded"."""
    agent = _Agent(llm_client=_BadClient())
    with pytest.raises(LLMDegradedError):
        _run(agent)
    assert agent.uses_llm is True, "a client is configured"
    assert agent.last_failure_reason


def _code_lines(module) -> list[str]:
    """Source with comments and docstrings stripped."""
    import io
    import tokenize
    from pathlib import Path

    source = Path(module.__file__).read_text(encoding="utf-8")
    out: list[str] = []
    prev_type = tokenize.INDENT
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and prev_type in (
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.NEWLINE,
            tokenize.NL,
        ):
            continue
        if tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            out.append(tok.string)
            prev_type = tok.type
    return out


def test_competition_analyzer_does_not_infer_from_uses_llm():
    import labpilot.research_engine.intelligence.analyzers.competition as mod

    assert "uses_llm" not in _code_lines(mod), (
        "competition.py must read last_generated_by, not uses_llm"
    )


_PROVENANCE_WRITERS = [
    "labpilot.research_engine.intelligence.fetch.enrich",
    "labpilot.research_engine.intelligence.brief.builder",
    "labpilot.research_engine.intelligence.knowledge.merger",
    "labpilot.research_engine.intelligence.hypothesis.assistant",
    "labpilot.research_engine.intelligence.analyzers.competition",
]


@pytest.mark.parametrize("module_path", _PROVENANCE_WRITERS)
def test_durable_writers_read_recorded_provenance(module_path):
    import importlib

    module = importlib.import_module(module_path)
    code = _code_lines(module)

    assert "last_generated_by" in code, f"{module_path} does not record provenance"
    assert "uses_llm" not in code, (
        f"{module_path} infers provenance from uses_llm, which is True even "
        "when the call failed"
    )


def test_degraded_still_records_provenance(monkeypatch):
    from labpilot.accessor.common.provenance import AgentInvocation, set_sink

    records: list[AgentInvocation] = []

    class _Sink:
        def record(self, invocation):
            records.append(invocation)

    set_sink(_Sink())
    try:
        with pytest.raises(LLMDegradedError):
            _run(_Agent(llm_client=_BadClient()))
    finally:
        set_sink(None)

    assert records and records[-1].failure_kind == "json_shape"


def test_unavailable_and_degraded_stay_distinct():
    with pytest.raises(LLMUnavailableError) as missing:
        _run(_Agent(llm_client=None))
    with pytest.raises(LLMDegradedError) as bad:
        _run(_Agent(llm_client=_BadClient()))
    assert not isinstance(missing.value, LLMDegradedError)
    assert "prov_probe" in str(bad.value)
