"""M14 phase 1 — every result records what actually produced it.

The rule throughout: assert the *recorded provenance*, never that a flag was
read. An artifact claiming `llm` for deterministic output is worse than one
claiming nothing, because it stops the reader checking.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from labpilot.accessor.common.micro_agents import BaseMicroAgent, StructuredContext


class _Out(BaseModel):
    value: str = "rule"


class _Agent(BaseMicroAgent):
    name = "prov_probe"
    output_model = _Out
    # No sleeping between retries in tests.
    llm_max_attempts = 1
    llm_retry_delay_seconds = 0.0

    def system_prompt(self) -> str:
        return "s"

    def user_prompt(self, context: StructuredContext) -> str:
        return "u"

    def _run_rule_engine(self, context: StructuredContext) -> _Out:
        return _Out(value="rule")


class _GoodClient:
    def complete(self, system, user, *, json_mode=False):
        return '{"value": "llm"}'


class _BadClient:
    def complete(self, system, user, *, json_mode=False):
        return "These rules outline the guidelines for participating..."


def _run(agent):
    return agent.run(StructuredContext())


# --- the three paths ---------------------------------------------------------


def test_llm_success_records_llm():
    agent = _Agent(llm_client=_GoodClient())
    result = _run(agent)
    assert result.value == "llm"
    assert agent.last_generated_by == "llm"
    assert agent.last_failure_reason is None


def test_llm_failure_records_rule_engine_and_a_reason():
    """The observed failure: a model answering a JSON prompt in prose."""
    agent = _Agent(llm_client=_BadClient())
    result = _run(agent)
    assert result.value == "rule"
    assert agent.last_generated_by == "rule_engine"
    assert agent.last_failure_reason
    assert "JSON" in agent.last_failure_reason


def test_no_client_records_rule_engine():
    """The blind spot: this path logs nothing at all, so the stamp is the only
    evidence the run was deterministic."""
    agent = _Agent(llm_client=None)
    result = _run(agent)
    assert result.value == "rule"
    assert agent.last_generated_by == "rule_engine"
    assert agent.last_failure_reason == "no llm client"


def test_provenance_resets_between_runs():
    """A successful run must not leave `llm` behind for a later failure."""
    agent = _Agent(llm_client=_GoodClient())
    _run(agent)
    assert agent.last_generated_by == "llm"

    agent.llm_client = _BadClient()
    _run(agent)
    assert agent.last_generated_by == "rule_engine", "stale provenance from prior run"


# --- the regression that motivated this --------------------------------------


def test_uses_llm_is_not_provenance():
    """`uses_llm` is "a client exists", not "the call succeeded".

    analyzers/competition.py recorded `page_enrichment_source = "llm"` from it,
    so a fallback run produced an artifact claiming the LLM had reasoned.
    """
    agent = _Agent(llm_client=_BadClient())
    _run(agent)
    assert agent.uses_llm is True, "a client is configured"
    assert agent.last_generated_by == "rule_engine", "but it did not produce this"
    assert agent.uses_llm != (agent.last_generated_by == "llm")


def _code_lines(module) -> list[str]:
    """Source with comments and docstrings stripped.

    A guard that greps raw text also matches the comment *explaining* the bug,
    which would make documenting it impossible.
    """
    import io
    import tokenize
    from pathlib import Path

    source = Path(module.__file__).read_text(encoding="utf-8")
    out: list[str] = []
    prev_type = tokenize.INDENT
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            continue
        # A STRING alone on a logical line is a docstring.
        if tok.type == tokenize.STRING and prev_type in (
            tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE, tokenize.NL,
        ):
            continue
        if tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            out.append(tok.string)
            prev_type = tok.type
    return out


def test_competition_analyzer_does_not_infer_from_uses_llm():
    """The file that had the bug must not reintroduce it — in *code*."""
    import labpilot.research_engine.intelligence.analyzers.competition as mod

    assert "uses_llm" not in _code_lines(mod), (
        "competition.py must read last_generated_by, not uses_llm — "
        "uses_llm is True even when the run fell back"
    )


# --- suite-wide coverage -----------------------------------------------------

#: Modules that stamp provenance onto a durable record. Adding a writer without
#: adding it here is what let three field names diverge in the first place.
_PROVENANCE_WRITERS = [
    "labpilot.research_engine.intelligence.fetch.enrich",
    "labpilot.research_engine.intelligence.brief.builder",
    "labpilot.research_engine.intelligence.knowledge.merger",
    "labpilot.research_engine.intelligence.hypothesis.assistant",
    "labpilot.research_engine.intelligence.analyzers.competition",
]


@pytest.mark.parametrize("module_path", _PROVENANCE_WRITERS)
def test_durable_writers_read_recorded_provenance(module_path):
    """No writer may re-derive provenance; all must read `last_generated_by`."""
    import importlib

    module = importlib.import_module(module_path)
    code = _code_lines(module)

    assert "last_generated_by" in code, f"{module_path} does not record provenance"
    assert "uses_llm" not in code, (
        f"{module_path} infers provenance from uses_llm, which is True even "
        "when the run fell back"
    )
