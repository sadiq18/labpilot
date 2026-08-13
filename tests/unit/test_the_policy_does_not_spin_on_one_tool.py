"""Re-running a tool that tells us nothing new must not be on offer.

Measured on rogii 2026-08-12, a full campaign spent on one tool:

    D-284: query_memory — The current MSE (194.8) is significantly higher than the
    D-285: query_memory — The current MSE (194.8) is far above the target of 5.
    ...
    D-291: query_memory — The current MSE (194.8) is still far above the target of
    D-292: — stop — stop:failing — 0 consecutive failed execution(s), 8 step(s)

Eight decisions, zero experiments, stopped on stagnation. `query_memory` does
not change the observation, so every step re-derived the same reasons from the
same numbers and reached the same answer. The offline policy has refused
non-`_REPEATABLE` repeats since S-019; the LLM path had no such rule.

The assertions read the allowlist the model is actually *sent*, not the set
passed in — narrowing that the prompt never sees would change nothing.
"""

from __future__ import annotations

import json

import pytest

from labpilot.research_engine.conductor.policy import _REPEATABLE, llm_next_action

ALLOWLIST = {"query_memory", "generate_plan", "run_plan", "reflect", "analyze_competition"}


class _Records:
    """Policy client that answers `tool` and keeps the allowlist it was sent."""

    def __init__(self, tool: str = "generate_plan") -> None:
        self.tool = tool
        self.seen: list[list[str]] = []

    def complete(self, system, user):  # noqa: ANN001
        self.seen.append(list(json.loads(user).get("allowlist") or []))
        return json.dumps({"tool": self.tool, "args": {}, "rationale": "r", "stop": False})


def _offered(
    completed: list[str], allowlist: set[str] | None = None, *, answers: str = "generate_plan"
) -> list[str]:
    """The allowlist the model is sent. `answers` must be a tool it may pick:
    an unavailable one routes into the gated-retry loop and asks the operator.
    """
    client = _Records(answers)
    llm_next_action({"completed_tools": completed}, set(allowlist or ALLOWLIST), client)
    return client.seen[0]


def test_the_tool_that_just_ran_is_not_offered_again() -> None:
    """The spin itself: eight identical `query_memory` choices."""
    assert "query_memory" not in _offered(["analyze_competition", "query_memory"])


@pytest.mark.parametrize("tool", sorted(_REPEATABLE))
def test_a_tool_that_pays_off_on_a_repeat_is_still_offered(tool: str) -> None:
    """`run_plan` twice is a campaign working, not a campaign stuck."""
    assert tool in _offered(["analyze_competition", tool])


def test_only_the_immediately_preceding_tool_is_ruled_out() -> None:
    """Otherwise a tool used once is dead for the rest of the campaign — and
    `query_memory` after new evidence is exactly the right move."""
    assert "query_memory" in _offered(["query_memory", "run_plan"])


def test_an_allowlist_is_never_emptied() -> None:
    """With nothing on offer the model can only stop, which turns a spin into a
    dead campaign — a worse failure than the one being fixed."""
    assert _offered(["query_memory"], {"query_memory"}, answers="query_memory") == ["query_memory"]


def test_a_first_step_is_unaffected() -> None:
    assert sorted(ALLOWLIST) == _offered([])
