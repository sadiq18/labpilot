"""The Conductor's execution path must receive the LLM client.

Measured 2026-08-07, first campaign after M10 routing landed. `research doctor`
reported `codegen -> groq-llama70b`, and the campaign still died at step 5:

    Task failed: CodeEngineerAgent requires an LLM and none is configured.
    Conductor stop: Code execution tools (run_experiment, run_plan, implement)
    cannot proceed ...

The cause: `decide_next` (the policy) was passed `llm_client`, but `Scheduler`
was constructed without one and dispatched tools with `**task.args` alone. Task
args are persisted JSON and cannot carry a live client, so `run_plan` built a
`CodeEngineeringCapability(llm_client=None)`.

Worth recording *why this was invisible before*: M14 phase 2a turned "no client"
from a silent degrade into a refusal. Previously this path fell through to the
rule engine, which returns `files=[]`, which lands in the template fallback —
the same baseline for every hypothesis. It is one of the mechanisms behind MSE
194.80 repeating twelve times, and it only became visible because the system was
made to fail loudly.
"""

from __future__ import annotations

from typing import Any

import pytest

from labpilot.research_engine.conductor.models import OsTask
from labpilot.research_engine.conductor.scheduler import Scheduler
from labpilot.research_engine.tools.descriptors import ToolResult


class _Recorder:
    """Registry double that records what the handler was invoked with."""

    def __init__(self, accepts: bool) -> None:
        self.seen: dict[str, Any] = {}
        if accepts:

            def handler(workspace, llm_client=None, **kw):  # noqa: ANN001
                self.seen = {"llm_client": llm_client, **kw}
                return ToolResult(refs=[])
        else:

            def handler(workspace, **kw):  # noqa: ANN001
                self.seen = dict(kw)
                return ToolResult(refs=[])

        self.handler = handler
        self.descriptor = type("D", (), {"handler": handler})()

    def get(self, name):  # noqa: ANN001
        return self.descriptor

    def require(self, name):  # noqa: ANN001
        return self.descriptor

    def invoke(self, name, workspace, **params):  # noqa: ANN001
        return self.handler(workspace, **params)


class _Store:
    def update_task_status(self, *a, **kw) -> None:  # noqa: ANN001, ANN002
        return None


def _task() -> OsTask:
    return OsTask(
        id="T-1",
        session_id="S-1",
        tool_name="run_plan",
        args={"plan_id": "P-001"},
    )


def test_client_reaches_a_handler_that_declares_it():
    """The bug: without this the campaign stops at the first code execution."""
    registry = _Recorder(accepts=True)
    sentinel = object()
    Scheduler(_Store(), registry, workspace=None, llm_client=sentinel).dispatch(_task())

    assert registry.seen["llm_client"] is sentinel
    assert registry.seen["plan_id"] == "P-001", "task args must survive injection"


def test_handler_without_the_parameter_is_untouched():
    """Injecting blindly would raise TypeError on every tool that does not take
    a client — so the check is signature-driven, not a tool allow-list."""
    registry = _Recorder(accepts=False)
    Scheduler(_Store(), registry, workspace=None, llm_client=object()).dispatch(_task())

    assert "llm_client" not in registry.seen
    assert registry.seen["plan_id"] == "P-001"


def test_no_client_configured_changes_nothing():
    """`--offline` passes None; that must stay None rather than becoming a key
    the handler then treats as configured."""
    registry = _Recorder(accepts=True)
    Scheduler(_Store(), registry, workspace=None, llm_client=None).dispatch(_task())
    assert registry.seen["llm_client"] is None


def test_explicit_task_arg_wins():
    """A client already in task args is not overwritten by the scheduler's."""
    registry = _Recorder(accepts=True)
    explicit = object()
    task = _task()
    task.args["llm_client"] = explicit
    Scheduler(_Store(), registry, workspace=None, llm_client=object()).dispatch(task)
    assert registry.seen["llm_client"] is explicit


@pytest.mark.parametrize(
    "tool", ["run_plan", "run_experiment", "implement", "generate_plan", "query_memory"]
)
def test_real_execution_tools_declare_the_parameter(tool):
    """Injection is signature-driven, so a tool that stops declaring
    `llm_client` would silently start running without one."""
    import inspect

    from labpilot.cli.config_helpers import default_tools

    descriptor = default_tools().get(tool)
    assert descriptor is not None, f"{tool} missing from the default registry"
    params = inspect.signature(descriptor.handler).parameters
    assert "llm_client" in params, f"{tool} would run without an LLM client"
