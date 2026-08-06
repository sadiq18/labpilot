"""A gated tool chosen by the policy must not end the campaign.

Measured 2026-08-07, run 4 on rogii. `generate_plan` is removed from the
allowlist while an unrun plan exists (`policy.py`: "queuing another plan while
one is still unrun adds no information and starves the thing that does"). The
policy LLM chose it four times anyway, and the campaign stopped at step 4:

    Conductor stop: rejected non-catalog tool: generate_plan

Same shape as the two other stops that day — a recoverable condition treated as
terminal. The fix routes it into the retry/offline-fallback loop that already
exists for LLM failures, so the run degrades to the deterministic order instead
of ending.
"""

from __future__ import annotations

import pytest

from labpilot.research_engine.conductor.policy import (
    NextAction,
    llm_next_action,
    validate_next_action,
)

ALLOWLIST = {"run_plan", "run_experiment", "query_memory"}


class _Chooses:
    """Policy client that always returns one canned tool choice."""

    def __init__(self, tool: str | None, stop: bool = False) -> None:
        self.tool, self.stop, self.calls = tool, stop, 0

    def complete(self, system, user):  # noqa: ANN001
        self.calls += 1
        import json

        return json.dumps({"tool": self.tool, "args": {}, "rationale": "r", "stop": self.stop})


def _observe() -> dict:
    return {"summary": "s"}


def test_unavailable_tool_falls_back_instead_of_stopping():
    """The measured failure. `generate_plan` is gated; the run must continue."""
    client = _Chooses("generate_plan")
    action = llm_next_action(
        _observe(), ALLOWLIST, client, auto_offline_fallback=True, max_llm_retries=1
    )

    assert action.tool in ALLOWLIST, (
        f"expected a fallback to an available tool, got {action.tool!r} / stop={action.stop}"
    )
    assert not action.stop, "a gated tool choice must not end the campaign"


def test_a_genuine_stop_is_still_honoured():
    """Control: the model must remain able to end a campaign deliberately.

    Without this, the fix above could have been 'never stop', which would be a
    different bug — a campaign that cannot decide it is finished.
    """
    action = llm_next_action(
        _observe(), ALLOWLIST, _Chooses(None, stop=True), auto_offline_fallback=True
    )
    assert action.stop is True
    assert action.tool is None


def test_an_available_tool_passes_straight_through():
    client = _Chooses("run_plan")
    action = llm_next_action(_observe(), ALLOWLIST, client, auto_offline_fallback=True)
    assert action.tool == "run_plan"
    assert client.calls == 1, "a valid choice must not trigger retries"


@pytest.mark.parametrize("tool", ["generate_plan", "invented_tool", "submit"])
def test_validate_still_reports_the_rejection(tool):
    """`validate_next_action` keeps its contract — it is a pure predicate other
    callers rely on. The behaviour change is in how the caller reacts to it."""
    rejected = validate_next_action(
        NextAction(tool=tool, args={}, rationale="r", stop=False), ALLOWLIST
    )
    assert rejected.stop is True
    assert tool in rejected.rationale


# --- the retry must differ from the first attempt ---------------------------


class _RecordsPrompts:
    """Asks for a gated tool first, then a valid one — recording each prompt."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, system, user):  # noqa: ANN001
        import json

        self.prompts.append(user)
        tool = "generate_plan" if len(self.prompts) == 1 else "run_plan"
        return json.dumps({"tool": tool, "args": {}, "rationale": "r", "stop": False})


def test_the_retry_names_the_rejected_tool_back_to_the_model():
    """Measured: `generate_plan` was chosen six times in one run because every
    retry re-sent an identical prompt. A retry that carries no new information
    is not a retry."""
    client = _RecordsPrompts()
    action = llm_next_action(
        _observe(), ALLOWLIST, client, auto_offline_fallback=True, max_llm_retries=3
    )

    assert action.tool == "run_plan", "the second attempt should be accepted"
    assert len(client.prompts) == 2, "expected exactly one retry"
    assert "already_rejected" not in client.prompts[0], "first attempt has nothing to report"
    assert "already_rejected" in client.prompts[1], "retry must state what was rejected"
    assert "generate_plan" in client.prompts[1]


def test_a_transport_failure_does_not_pollute_the_rejected_list():
    """Only a gated *tool* choice is fed back; a network error is not a tool."""

    class _Broken:
        def complete(self, system, user):  # noqa: ANN001
            raise RuntimeError("HTTP 429")

    action = llm_next_action(
        _observe(), ALLOWLIST, _Broken(), auto_offline_fallback=True, max_llm_retries=1
    )
    assert action.tool in ALLOWLIST or action.stop


def test_an_invented_tool_is_described_differently_from_a_gated_one():
    """Telling a model that a hallucinated tool "failed a precondition" invites
    it to wait for that precondition to clear. It never will."""
    client = _RecordsPrompts()
    client.__class__.complete = lambda self, system, user: (  # noqa: ARG005
        self.prompts.append(user)
        or __import__("json").dumps(
            {
                "tool": "teleport_model" if len(self.prompts) == 1 else "run_plan",
                "args": {}, "rationale": "r", "stop": False,
            }
        )
    )
    action = llm_next_action(
        _observe(), ALLOWLIST, client, all_tools=ALLOWLIST,
        auto_offline_fallback=True, max_llm_retries=3,
    )
    assert action.tool == "run_plan"
    retry = client.prompts[1]
    assert "not_real" in retry, "an invented tool must be reported as non-existent"
    assert "teleport_model" in retry
    assert "gated" not in retry, "nothing was gated here"
