"""`implement` must have something to implement, and must say when it did not.

Measured on rogii 2026-08-09. With `run_plan`, `run_experiment` and
`generate_plan` all correctly closed, the campaign spent **all eight steps** on
`implement` — 16 dispatches, 5 recorded `completed`, and `pipeline/train.py`
untouched throughout.

Two defects, compounding:

* the tool was **ungated**, so it stayed available with no plan and no
  hypothesis to implement — the one door left open when every other closed;
* it reported **success for a no-op**, so the policy kept choosing it. From its
  side the tool had just worked.

Only `steps_since_success` could see this: nothing *failed*, so
`consecutive_failures` stayed at 0 for the whole run.
"""

from __future__ import annotations

import pytest

from labpilot.research_engine.tools.handlers.specialists import (
    ImplementProducedNothingError,
    implement,
)


class _Agent:
    def __init__(self, refs):
        self._refs = refs


def _patch_specialist(monkeypatch, refs):
    """Install a specialist returning `refs`, bypassing the real agent."""
    import labpilot.research_engine.tools.handlers.specialists as mod

    class _Candidate:
        name = "fake"
        agent = _Agent(refs)

    class _Registry:
        def candidates(self, *, capability):
            return [_Candidate()]

    monkeypatch.setattr(mod, "build_default_specialist_registry", lambda **k: _Registry())
    monkeypatch.setattr(mod, "execute_agent_sync", lambda *a, **k: refs)
    monkeypatch.setattr(mod, "_bundle", lambda *a, **k: None)


def _ref(path):
    """A real ArtifactRef — ToolResult validates its refs, so a stand-in would
    only prove the stand-in."""
    from labpilot.research_engine.artifacts.base import ArtifactRef

    return ArtifactRef(kind="code", id=path, schema_id="code/v1", path=path, competition="demo")


def test_writing_nothing_is_a_failure_not_a_success(monkeypatch, tmp_path):
    """The exact rogii behaviour: five `completed` tasks, no file changed."""
    _patch_specialist(monkeypatch, [])

    with pytest.raises(ImplementProducedNothingError):
        implement(object(), description="add catboost")


def test_the_failure_says_why(monkeypatch):
    """A campaign that stops must say what it was unable to do."""
    _patch_specialist(monkeypatch, [])

    with pytest.raises(ImplementProducedNothingError, match="no files"):
        implement(object(), description="add catboost")


def test_writing_files_still_succeeds(monkeypatch):
    """The carve-out must not cost the behaviour it guards."""
    _patch_specialist(monkeypatch, [_ref("pipeline/train.py")])

    result = implement(object(), description="add catboost")

    assert result.data["paths"] == ["pipeline/train.py"]


def test_it_is_gated_on_a_runnable_plan(monkeypatch, tmp_path):
    """With no plan there is nothing to implement, and leaving it open made it
    an escape hatch from the absence of every other tool."""
    import labpilot.research_engine.conductor.policy as policy_mod

    monkeypatch.setattr(policy_mod, "has_runnable_plan", lambda ws: False)
    monkeypatch.setattr(policy_mod, "has_unrun_plan", lambda ws: False)
    monkeypatch.setattr(policy_mod, "viable_hypothesis_count", lambda kd, c: 9)
    monkeypatch.setattr(policy_mod, "hours_since_last_artifact", lambda ws: 2.0)

    class _WS:
        knowledge_dir = tmp_path
        competition = "demo"
        root = tmp_path
        layout = "workspace"

    tools = policy_mod.available_tools(_WS(), {"implement", "run_plan", "generate_plan"})

    assert "implement" not in tools
    assert "run_plan" not in tools


def test_it_is_offered_when_a_plan_exists(monkeypatch, tmp_path):
    import labpilot.research_engine.conductor.policy as policy_mod

    monkeypatch.setattr(policy_mod, "has_runnable_plan", lambda ws: True)
    monkeypatch.setattr(policy_mod, "has_unrun_plan", lambda ws: False)
    monkeypatch.setattr(policy_mod, "viable_hypothesis_count", lambda kd, c: 9)
    monkeypatch.setattr(policy_mod, "hours_since_last_artifact", lambda ws: 2.0)

    class _WS:
        knowledge_dir = tmp_path
        competition = "demo"
        root = tmp_path
        layout = "workspace"

    assert "implement" in policy_mod.available_tools(_WS(), {"implement"})


def test_implement_and_run_plan_disappear_together(monkeypatch, tmp_path):
    """Both act on a plan, so neither should survive the other's absence."""
    import labpilot.research_engine.conductor.policy as policy_mod

    monkeypatch.setattr(policy_mod, "has_unrun_plan", lambda ws: False)
    monkeypatch.setattr(policy_mod, "viable_hypothesis_count", lambda kd, c: 9)
    monkeypatch.setattr(policy_mod, "hours_since_last_artifact", lambda ws: 2.0)

    class _WS:
        knowledge_dir = tmp_path
        competition = "demo"
        root = tmp_path
        layout = "workspace"

    for runnable in (True, False):
        monkeypatch.setattr(policy_mod, "has_runnable_plan", lambda ws, r=runnable: r)
        tools = policy_mod.available_tools(_WS(), {"implement", "run_plan"})
        assert ("implement" in tools) == ("run_plan" in tools)
