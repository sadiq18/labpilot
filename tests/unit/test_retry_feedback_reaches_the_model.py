"""A retry that cannot see why the last attempt failed rebuilds blind.

The channel exists end to end: `Engineer._first_failure_reason` computes the
reason, the code-engineering capability threads it into context data as
`retry_reason`, and `code_engineer_user.j2` has a prominent block for it —

    The previous attempt at this file FAILED. Fix the cause before anything
    else — regenerating the same approach will fail the same way.

`CodeEngineerAgent.user_prompt` never passed it to the template. Jinja renders
an undefined name as empty, so the block was always skipped and every retry
re-sent the original prompt verbatim.

Measured 2026-08-20: three consecutive attempts wrote `pipeline/baseline.py`
instead of the entry point, and three before them omitted the same PEP 723
dependencies. Each retry was blind to a message naming its cause exactly, and
each cost a campaign step — the breaker stops at three, so one unreadable
failure consumes the whole allowance.

`engineer.py`'s own docstring already named this shape: *"`retry_reason` stayed
empty and three consecutive retries rebuilt blind while the error sat one field
away."*
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from labpilot.research_engine.execution.micro_agents.code_engineer.agent import (
    CodeEngineerAgent,
)

_REASON = "pipeline/train.py missing after apply — the proposal wrote baseline.py"


def _context(**data: object) -> SimpleNamespace:
    base = {
        "task_id": "P-001-T03",
        "task_type": "write_code",
        "task_description": "Write the baseline training script",
        "plan_id": "P-001",
        "problem_type": "tabular_regression",
    }
    return SimpleNamespace(competition="demo", question="reach rmse 2.0", text="", data=base | data)


def test_the_retry_reason_reaches_the_prompt() -> None:
    prompt = CodeEngineerAgent().user_prompt(_context(retry_reason=_REASON))

    assert _REASON in prompt
    assert "FAILED" in prompt, "the block that frames the reason must render too"


def test_a_first_attempt_carries_no_failure_notice() -> None:
    """The block must stay out of the way when there is nothing to report —
    telling a fresh attempt that it already failed is its own kind of wrong."""
    prompt = CodeEngineerAgent().user_prompt(_context())

    assert "The previous attempt at this file FAILED" not in prompt


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_a_blank_reason_is_not_a_failure_notice(blank: object) -> None:
    prompt = CodeEngineerAgent().user_prompt(_context(retry_reason=blank))

    assert "The previous attempt at this file FAILED" not in prompt


def test_a_long_reason_is_bounded() -> None:
    """A failure excerpt is a hint, not a log. Unbounded, a stack trace would
    crowd out the profile and inventory the model needs to write the file."""
    prompt = CodeEngineerAgent().user_prompt(_context(retry_reason="x" * 10_000))

    assert "x" * 2_000 in prompt
    assert "x" * 2_001 not in prompt
