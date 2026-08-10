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

**Three places execute model-written code, not two.** Reviewing this change
found the third: `pip install -r requirements.txt` builds packages a model
named, running their `setup.py` with the operator's keys. Covering two of three
and calling the criterion done was the mistake — so the installer is here too.

And all three needed a *bound*, not only a stripped environment. The unit gate
had no `timeout` at all while its sibling smoke gate had one; the installer had
neither. A timeout alone would have been the wrong fix: `subprocess.run` raises
`TimeoutExpired`, `capability.execute` is not wrapped by the engine, and an
exception produces no evidence file and no verdict — the silent failure this
milestone exists to remove, arriving through the fix for a hang.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from helpers.capability_context import capability_context

from labpilot.research_engine.execution.capabilities.dependency import (
    DependencyCapability,
)
from labpilot.research_engine.execution.capabilities.dependency import (
    capability as dependency_module,
)
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


def _capture(monkeypatch, module=None) -> dict:
    """Record the argv, environment and bound the gate hands its subprocess."""
    seen: dict = {}

    def _fake_run(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        seen["env"] = kwargs.get("env")
        seen["cwd"] = kwargs.get("cwd")
        seen["timeout"] = kwargs.get("timeout")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr((module or verification_module).subprocess, "run", _fake_run)
    return seen


def _launched(seen: dict) -> dict:
    """The captured call, or a legible failure if the gate never made one.

    Every assertion below reads a key this dict only has once the gate reached
    `subprocess.run`, and `_smoke` has three earlier returns — no `train.py`, a
    syntax error, and dry-run/`smoke_syntax_only`. Subscripting straight into it
    turned "the gate stopped early" into `KeyError: 'cmd'`, which says nothing
    about what broke. Reported reviewing PR #124, and the same unguarded-subscript
    shape as PR #121's round-8 finding.
    """
    assert seen, "the gate returned before launching anything; there is no call to inspect"
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
    assert _launched(seen)["cmd"] == training_command(train, python=sys.executable)


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

    assert _launched(seen)["env"] is not None, "inheriting the parent environment is the defect"
    leaked = sorted(name for name in _CREDENTIALS if name in _launched(seen)["env"])
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

    assert _launched(seen)["env"] == expected


def test_the_smoke_gate_still_tells_the_script_it_is_a_smoke_run(tmp_path, monkeypatch):
    """Guards the fix rather than the defect. Swapping in `child_environment()`
    would silently drop `LABPILOT_SMOKE` — a template that shortens its run when
    it sees the flag would quietly start doing full training inside the gate.
    """
    context = _smoke_context(tmp_path, monkeypatch)
    seen = _capture(monkeypatch)

    VerificationCapability().execute(context)

    assert _launched(seen)["env"]["LABPILOT_SMOKE"] == "1"


def test_the_unit_gate_is_not_told_it_is_a_smoke_run(tmp_path, monkeypatch):
    """The flag belongs to one gate. Setting it for both would make every
    generated test suite take the shortened path a smoke run asks for."""
    context = _unit_context(tmp_path, monkeypatch)
    seen = _capture(monkeypatch)

    VerificationCapability().execute(context)

    assert "LABPILOT_SMOKE" not in _launched(seen)["env"]


# -- bounds: every gate that runs model-written code says when to stop ---------


#: What the child managed to print before it was killed. **Bytes**, because that
#: is what `TimeoutExpired` carries on POSIX even when `text=True` was passed —
#: the exception comes from the inner `communicate()`, before decoding. A handler
#: that interpolates it writes a literal `b'...'` into the log.
_PARTIAL_STDOUT = b"collected 3 items\nrunning test_alpha\n"
_PARTIAL_STDERR = b"warning: slow fixture\n"


def _timeout_after(monkeypatch, module, *, partial: bool = True) -> None:
    """Make the gate's subprocess exceed its own limit, as the real one does."""

    def _hangs(cmd, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd,
            kwargs.get("timeout") or 0,
            output=_PARTIAL_STDOUT if partial else None,
            stderr=_PARTIAL_STDERR if partial else None,
        )

    monkeypatch.setattr(module.subprocess, "run", _hangs)


@pytest.mark.parametrize(
    ("gate", "default"), [("smoke", 120), ("unit", 600)], ids=["smoke", "unit"]
)
def test_each_gate_bounds_how_long_it_waits(tmp_path, monkeypatch, gate, default):
    """The unit gate had no `timeout` and its sibling smoke gate did, so a
    generated `while True:` blocked the campaign with no verdict, no evidence
    and no failure — while the same file under the smoke gate returned in two
    minutes. Reported reviewing PR #124.

    600s for unit rather than smoke's 120: a real generated suite legitimately
    runs longer than a smoke check, and the bound is there to catch a hang, not
    to police slowness.
    """
    context = (_smoke_context if gate == "smoke" else _unit_context)(tmp_path, monkeypatch)
    seen = _capture(monkeypatch)

    VerificationCapability().execute(context)

    assert _launched(seen)["timeout"] == default


@pytest.mark.rejects("verification:timeout")
@pytest.mark.parametrize("gate", ["smoke", "unit"])
def test_a_gate_that_times_out_returns_a_verdict_rather_than_raising(tmp_path, monkeypatch, gate):
    """A bound with no handler trades a hang for a vanish.

    `subprocess.run` raises `TimeoutExpired`, and `capability.execute` is called
    unwrapped at `engineer.py:227` — so the exception escapes, no evidence file
    is written, and the step leaves no record of having decided anything. That
    is worse than the hang it replaces, and exactly what M20 is about: the gate
    has to *say no*, not disappear.

    The smoke gate has carried a `timeout` and no handler since before this
    branch, so this covers a live gap as well as the new one.
    """
    context = (_smoke_context if gate == "smoke" else _unit_context)(tmp_path, monkeypatch)
    _timeout_after(monkeypatch, verification_module)

    result = VerificationCapability().execute(context)

    assert result.passed is False
    assert "timeout" in result.checks
    assert "timed out" in (result.error or "").lower()


# -- the third place model-written code runs: the installer --------------------


def _install_context(tmp_path, monkeypatch):
    """A workspace whose requirements name a package that is not installed, so
    the gate reaches `pip install` rather than the already-satisfied return."""
    for name, value in _CREDENTIALS.items():
        monkeypatch.setenv(name, value)
    context = capability_context(tmp_path, task_type=TaskType.INSTALL_PACKAGE)
    (context.workspace_root / "requirements.txt").write_text(
        "labpilot-not-a-real-package==1.0\n", encoding="utf-8"
    )
    return context


def test_the_installer_does_not_hand_credentials_to_the_packages_it_builds(tmp_path, monkeypatch):
    """Reported reviewing PR #124: two of the three places that execute
    model-written code were fixed, and the criterion marked done.

    Installing a package **runs** it — `setup.py` or a PEP 517 backend executes
    during the build. The requirements file is model-writable, codegen chooses
    the paths it writes, and `install=True` is the production default, so a
    typo-squatted name is enough to run arbitrary code holding every key
    `child_environment` exists to withhold. This was the worst of the three and
    the one left out.
    """
    context = _install_context(tmp_path, monkeypatch)
    seen = _capture(monkeypatch, dependency_module)

    DependencyCapability(install=True).execute(context)

    assert _launched(seen)["env"] is not None, "inheriting the parent environment is the defect"
    leaked = sorted(name for name in _CREDENTIALS if name in _launched(seen)["env"])
    assert not leaked, f"pip install handed the packages it builds: {leaked}"
    assert _launched(seen)["env"] == child_environment()


def test_the_installer_bounds_how_long_a_build_may_take(tmp_path, monkeypatch):
    """Longer than either gate, because a source build of a large wheel is slow
    and being killed mid-build is a worse failure than waiting. The bound exists
    so a package that hangs on a prompt or a dead index cannot stall the
    campaign for good."""
    context = _install_context(tmp_path, monkeypatch)
    seen = _capture(monkeypatch, dependency_module)

    DependencyCapability(install=True).execute(context)

    assert _launched(seen)["timeout"] == 900


@pytest.mark.rejects("dependency:timeout")
def test_an_install_that_times_out_returns_a_verdict_rather_than_raising(tmp_path, monkeypatch):
    """Same reason as the gates: an exception out of `execute` writes no
    evidence and records no decision."""
    context = _install_context(tmp_path, monkeypatch)
    _timeout_after(monkeypatch, dependency_module)

    result = DependencyCapability(install=True).execute(context)

    assert result.passed is False
    assert "timeout" in result.checks
    assert "timed out" in (result.error or "").lower()


# -- a timeout is still a report ----------------------------------------------


@pytest.mark.parametrize("gate", ["smoke", "unit"])
def test_a_gate_that_times_out_keeps_what_the_process_managed_to_say(tmp_path, monkeypatch, gate):
    """Reported reviewing PR #124, round 2.

    The first version of `_timed_out` wrote its own one-line message and dropped
    `expired.output`, so `logs/unit_tests.log` read `pytest timed out after 600s`
    and nothing else — while the success path writes returncode, stdout and
    stderr to that same file. The failing case was the thinner record, which is
    the asymmetry PR #121 fixed in `evaluation._infer` reappearing in the handler
    written to stop a different silence.

    The partial output is the whole diagnosis: it names the test that was running
    when the clock ran out.
    """
    context = (_smoke_context if gate == "smoke" else _unit_context)(tmp_path, monkeypatch)
    _timeout_after(monkeypatch, verification_module)

    result = VerificationCapability().execute(context)

    log = Path(result.paths[0]).read_text(encoding="utf-8")
    assert "running test_alpha" in log
    assert "warning: slow fixture" in log
    assert "running test_alpha" in (result.error or "")


@pytest.mark.parametrize("gate", ["smoke", "unit"])
def test_a_timeout_report_is_text_not_a_bytes_repr(tmp_path, monkeypatch, gate):
    """`TimeoutExpired.output` is bytes on POSIX **even when `text=True` was
    passed** — the exception is raised by the inner `communicate()` before
    decoding. Interpolating it puts `b'collected 3 items\\n...'` in the log, which
    is worse than useless: it looks like a record and reads like an escape
    sequence."""
    context = (_smoke_context if gate == "smoke" else _unit_context)(tmp_path, monkeypatch)
    _timeout_after(monkeypatch, verification_module)

    result = VerificationCapability().execute(context)

    log = Path(result.paths[0]).read_text(encoding="utf-8")
    assert "\\n" not in log, "an escaped newline means bytes were interpolated"
    assert "b'" not in log and 'b"' not in log


@pytest.mark.parametrize("gate", ["smoke", "unit"])
def test_a_gate_that_times_out_with_no_output_still_reports(tmp_path, monkeypatch, gate):
    """A process killed before it printed anything leaves `output=None`. The
    handler must still produce its verdict rather than fail formatting it."""
    context = (_smoke_context if gate == "smoke" else _unit_context)(tmp_path, monkeypatch)
    _timeout_after(monkeypatch, verification_module, partial=False)

    result = VerificationCapability().execute(context)

    assert result.passed is False
    assert "timed out" in (result.error or "").lower()


def test_an_install_that_times_out_keeps_what_pip_managed_to_say(tmp_path, monkeypatch):
    """Same defect, same fix, in the installer. Without it a hung build names no
    package, though pip's partial output says which one it was collecting."""
    context = _install_context(tmp_path, monkeypatch)
    _timeout_after(monkeypatch, dependency_module)

    result = DependencyCapability(install=True).execute(context)

    assert "running test_alpha" in (result.error or "")
    assert "b'" not in (result.error or "")


# -- the bounds are actually overridable --------------------------------------


@pytest.mark.parametrize(
    ("gate", "key"),
    [("smoke", "smoke_timeout_s"), ("unit", "unit_timeout_s")],
    ids=["smoke", "unit"],
)
def test_a_gate_bound_can_be_overridden_by_its_constraint(tmp_path, monkeypatch, gate, key):
    """Reported reviewing PR #124, round 2: nothing read any of these keys, not
    even the pre-existing `smoke_timeout_s`, while the roadmap said the bounds
    were constraint-overridable. A renamed or mistyped key falls back to the
    default in silence, and every test that asserts the default still passes."""
    context = (_smoke_context if gate == "smoke" else _unit_context)(tmp_path, monkeypatch)
    context.constraints[key] = 7
    seen = _capture(monkeypatch)

    VerificationCapability().execute(context)

    assert _launched(seen)["timeout"] == 7


def test_the_install_bound_can_be_overridden_by_its_constraint(tmp_path, monkeypatch):
    context = _install_context(tmp_path, monkeypatch)
    context.constraints["install_timeout_s"] = 11
    seen = _capture(monkeypatch, dependency_module)

    DependencyCapability(install=True).execute(context)

    assert _launched(seen)["timeout"] == 11
