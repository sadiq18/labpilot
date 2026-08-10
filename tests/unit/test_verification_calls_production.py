"""M20 exit criterion 2: a check invokes production, it does not resemble it.

The defect that named this criterion: the smoke gate built its own command
instead of calling `training_command`, so the two drifted apart exactly where it
mattered. Measured on rogii 2026-08-08 — a `train.py` whose PEP 723 block was
unterminated passed smoke under a bare `python` and then failed training, where
uv refused the whole script.

That command is shared now. The *environment around it* was not, and it is the
half where the divergence is worse than a wrong verdict: `TrainingRunner` runs
generated code through `child_environment()`, which strips the operator's
provider and Kaggle credentials, and both verification gates ran the same
generated code with `os.environ` intact. `child_environment`'s own docstring
says that code "has no business holding the operator's provider keys or Kaggle
credentials" — and the gate that runs first handed it exactly those.

Asserted by capturing what the gate actually passes to `subprocess.run`, not by
reading the capability's source for the right call. M20's first criterion spent
seven review rounds inside a source parser before that lesson landed; a check on
what a run *did* cannot be fooled by a spelling nobody anticipated.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from helpers.capability_context import capability_context

from labpilot.research_engine.execution.capabilities.verification import (
    VerificationCapability,
)
from labpilot.research_engine.execution.capabilities.verification import (
    capability as verification_module,
)
from labpilot.research_engine.execution.training import environment
from labpilot.research_engine.execution.training.environment import (
    child_environment,
    training_command,
)
from labpilot.research_engine.planner.schemas.task_types import TaskType

#: Names `is_secret_env` recognises, one by prefix and one by marker, so a fix
#: that happened to catch only one form still fails this.
_CREDENTIALS = {"GROQ_API_KEY": "sk-groq-sentinel", "KAGGLE_KEY": "kaggle-sentinel"}

_RUNNABLE = "print('ok')\n"

#: A script that declares dependencies, which is the shape the criterion is
#: about: `training_command` answers `uv run --script` for this one and
#: `[python, path]` for a plain file. A fixture with no PEP 723 block makes the
#: two indistinguishable — a hand-built `[sys.executable, str(train)]` passed
#: the comparison below until a mutation sweep showed it, which is the rogii
#: 2026-08-08 failure surviving inside the test written to catch it.
_DECLARING = '# /// script\n# dependencies = ["pandas"]\n# ///\nprint("ok")\n'


def _capture(monkeypatch) -> dict:
    """Record the argv and environment the gate hands its subprocess."""
    seen: dict = {}

    def _fake_run(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        seen["env"] = kwargs.get("env")
        seen["cwd"] = kwargs.get("cwd")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(verification_module.subprocess, "run", _fake_run)
    return seen


def _smoke_context(tmp_path, monkeypatch, script: str = _RUNNABLE):
    for name, value in _CREDENTIALS.items():
        monkeypatch.setenv(name, value)
    context = capability_context(tmp_path, task_type=TaskType.RUN_SMOKE_TEST)
    (context.workspace_root / "pipeline" / "train.py").write_text(script, encoding="utf-8")
    return context


def _unit_context(tmp_path, monkeypatch):
    for name, value in _CREDENTIALS.items():
        monkeypatch.setenv(name, value)
    context = capability_context(tmp_path, task_type=TaskType.RUN_UNIT_TEST)
    tests_dir = context.workspace_root / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_generated.py").write_text("def test_ok():\n    pass\n", encoding="utf-8")
    return context


@pytest.mark.parametrize("script", [_RUNNABLE, _DECLARING], ids=["plain", "declares_deps"])
def test_the_smoke_gate_runs_the_command_production_runs(tmp_path, monkeypatch, script):
    """The criterion's original defect, kept red-able.

    Compared against `training_command` itself rather than against a literal
    argv, because a test that spelled out `["uv", "run", "--script", ...]` would
    be a second implementation of the thing under test — the very shape the
    criterion forbids.

    Both script shapes, because they are the same argv for a plain file and
    only diverge once dependencies are declared — which is where rogii's gate
    passed a script that training then refused. `uv_available` is pinned so the
    branch is chosen by the fixture rather than by what is installed on the
    machine running the suite.
    """
    monkeypatch.setattr(environment, "uv_available", lambda: True)
    context = _smoke_context(tmp_path, monkeypatch, script)
    seen = _capture(monkeypatch)

    VerificationCapability().execute(context)

    train = context.workspace_root / "pipeline" / "train.py"
    assert seen["cmd"] == training_command(train, python=sys.executable)


@pytest.mark.parametrize("gate", ["smoke", "unit"])
def test_neither_gate_hands_the_operator_credentials_to_generated_code(tmp_path, monkeypatch, gate):
    """Both gates run code nobody reviewed. `TrainingRunner` strips credentials
    before doing the same thing, and a check that is more permissive than the
    run it stands in for is not standing in for it.

    Red-then-green: before the fix the smoke gate passed `{**os.environ, ...}`
    and the unit gate passed no `env` at all, so both children inherited every
    key the operator had exported.
    """
    context = (_smoke_context if gate == "smoke" else _unit_context)(tmp_path, monkeypatch)
    seen = _capture(monkeypatch)

    VerificationCapability().execute(context)

    assert seen["env"] is not None, "inheriting the parent environment is the defect"
    leaked = sorted(name for name in _CREDENTIALS if name in seen["env"])
    assert not leaked, f"{gate} gate handed generated code: {leaked}"


@pytest.mark.parametrize("gate", ["smoke", "unit"])
def test_each_gate_builds_its_environment_from_the_one_production_uses(tmp_path, monkeypatch, gate):
    """Stronger than "no credentials": the environment *is* production's, so a
    later change to what counts as a secret reaches the gates without anyone
    remembering they exist.

    The smoke gate adds `LABPILOT_SMOKE`, which is the one difference it is
    entitled to — it is what a script reads to know it is being smoke-tested,
    and production deliberately does not set it.
    """
    context = (_smoke_context if gate == "smoke" else _unit_context)(tmp_path, monkeypatch)
    seen = _capture(monkeypatch)

    VerificationCapability().execute(context)

    expected = child_environment()
    if gate == "smoke":
        expected = {**expected, "LABPILOT_SMOKE": "1"}

    assert seen["env"] == expected


def test_the_smoke_gate_still_tells_the_script_it_is_a_smoke_run(tmp_path, monkeypatch):
    """Guards the fix rather than the defect. Swapping in `child_environment()`
    would silently drop `LABPILOT_SMOKE` — a template that shortens its run when
    it sees the flag would quietly start doing full training inside the gate.
    """
    context = _smoke_context(tmp_path, monkeypatch)
    seen = _capture(monkeypatch)

    VerificationCapability().execute(context)

    assert seen["env"]["LABPILOT_SMOKE"] == "1"


def test_the_unit_gate_is_not_told_it_is_a_smoke_run(tmp_path, monkeypatch):
    """The flag belongs to one gate. Setting it for both would make every
    generated test suite take the shortened path a smoke run asks for."""
    context = _unit_context(tmp_path, monkeypatch)
    seen = _capture(monkeypatch)

    VerificationCapability().execute(context)

    assert "LABPILOT_SMOKE" not in seen["env"]
