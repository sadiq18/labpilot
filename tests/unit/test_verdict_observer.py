"""The observer, held to the standard it exists to enforce.

This mechanism decides whether every other rejection test in the suite counts,
so the question it has to answer first is its own: *can it fail?* Each test
below is written from the failure it is meant to catch — a marker on a test that
rejects nothing, a marker naming the wrong gate, a recorder that keeps recording
after its block — rather than from the behaviour it is meant to allow.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from helpers.verdict_observer import Verdict, recording, satisfies, unearned

_REJECTED = Verdict(capability="reporting", checks=("reflect", "evidence_card"), passed=False)


def test_a_passing_verdict_never_satisfies_a_rejection_claim():
    """The whole point. A gate that ran and approved has not been shown to
    reject, and the marker says rejection."""
    approved = Verdict(capability="reporting", checks=("reflect",), passed=True)

    assert not satisfies(approved, "reporting")
    assert not satisfies(approved, "reporting:reflect")


def test_a_claim_is_not_satisfied_by_a_different_capability():
    assert not satisfies(_REJECTED, "submission")
    assert not satisfies(_REJECTED, "submission:reflect")


def test_a_named_check_must_actually_be_among_the_checks():
    """The granularity that makes `reporting:reflect` mean more than
    `reporting` — four gates report under `reporting`, and a test proving one
    of them must not be readable as proof of the other three."""
    assert satisfies(_REJECTED, "reporting:reflect")
    assert not satisfies(_REJECTED, "reporting:update_belief")


def test_a_bare_capability_claim_covers_a_verdict_with_no_checks():
    """Not every verdict carries a label. If the bare form did not match them,
    the gates hardest to name would be the ones nobody could claim."""
    unlabelled = Verdict(capability="workspace", checks=(), passed=False)

    assert satisfies(unlabelled, "workspace")
    assert not satisfies(unlabelled, "workspace:anything")


def test_unearned_reports_exactly_the_claims_nothing_backs():
    claims = ["reporting:reflect", "reporting:update_belief", "submission"]

    assert unearned(claims, [_REJECTED]) == ["reporting:update_belief", "submission"]
    assert unearned([], [_REJECTED]) == []


def test_recording_sees_a_real_capability_reject_and_stops_when_the_block_ends():
    """Observation has to survive the path production actually takes — through
    `evidence()` and the capability's own helpers, not a constructor a test
    calls directly."""
    from labpilot.research_engine.execution.schemas import TaskEvidence

    with recording() as observed:
        TaskEvidence(task_id="t", execution_id="e", capability="x", checks=["c"], passed=False)

    assert observed == [Verdict(capability="x", checks=("c",), passed=False)]

    TaskEvidence(task_id="t", execution_id="e", capability="y", checks=[], passed=False)
    assert len(observed) == 1, "the recorder outlived its block"


def test_nested_recording_restores_the_outer_recorder_rather_than_the_original():
    """A test that opens its own recorder must not silently switch the suite's
    autouse observation off for everything after it."""
    from labpilot.research_engine.execution.schemas import TaskEvidence

    with recording() as outer:
        with recording() as inner:
            TaskEvidence(task_id="t", execution_id="e", capability="inner", passed=False)
        TaskEvidence(task_id="t", execution_id="e", capability="after", passed=False)

    assert [v.capability for v in inner] == ["inner"]
    assert [v.capability for v in outer] == ["inner", "after"]


_TESTS = Path(__file__).resolve().parents[1]

_FULL_CONFTEST = """
from helpers.verdict_observer import (  # noqa: F401
    pytest_runtest_call,
    pytest_runtest_makereport,
    verdict_observer,
)
"""


def _run_pytest(tmp_path: Path, body: str, conftest: str = _FULL_CONFTEST):
    """A real pytest process over a generated test file.

    A subprocess rather than `pytester` because what is under test is the
    plugin's effect on *reporting* — outcomes, counts, and whether the session
    survives — and that is what a run prints.
    """
    (tmp_path / "conftest.py").write_text(
        textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(_TESTS)!r})

        def pytest_configure(config):
            config.addinivalue_line("markers", "rejects: M20")
        """)
        + textwrap.dedent(conftest),
        encoding="utf-8",
    )
    (tmp_path / "test_generated.py").write_text(textwrap.dedent(body), encoding="utf-8")
    # Run *from* tmp_path against `.`, so node ids are short relative paths.
    # pytest truncates each summary line to the terminal width, and with an
    # absolute tmp path in it the test name is what gets cut — a difference
    # between this machine and CI rather than between pass and fail. `--tb=no`
    # for the same reason: without it every reason is printed twice, once in the
    # traceback block and once in the summary. Both learned from CI failing a
    # commit that was green here.
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "--tb=no", "."],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "COLUMNS": "200"},
    )


def _failures(result) -> list[str]:
    """The short-summary line per failed test: `FAILED <id> - <reason>`."""
    return [line for line in result.stdout.splitlines() if line.startswith("FAILED")]


_LYING_TEST = """
import pytest
from labpilot.research_engine.execution.schemas import TaskEvidence

@pytest.mark.rejects("reporting:reflect")
def test_claims_a_rejection_it_never_causes():
    assert True

@pytest.mark.rejects("reporting:reflect")
def test_causes_a_rejection_under_a_different_check():
    TaskEvidence(task_id="t", execution_id="e", capability="reporting",
                 checks=["generate_report"], passed=False)

@pytest.mark.rejects("reporting:reflect")
def test_causes_the_rejection_it_claims():
    TaskEvidence(task_id="t", execution_id="e", capability="reporting",
                 checks=["reflect"], passed=False)
"""


@pytest.mark.slow
def test_a_marker_the_run_does_not_back_fails_that_test(tmp_path: Path):
    """The wiring, end to end, in a real pytest process.

    Everything above tests the predicate; this tests that the predicate is
    actually attached to the marker. Under the mechanism this replaces, a
    marker was a comment — the seven review rounds were spent on a parser that
    read test *sources* to decide which markers existed, and never on whether a
    marked test rejected anything. Here the first two of these three tests fail,
    and their bodies pass.
    """
    result = _run_pytest(tmp_path, _LYING_TEST)

    assert "1 passed" in result.stdout and "2 failed" in result.stdout, result.stdout
    assert "test_claims_a_rejection_it_never_causes" in result.stdout
    assert "test_causes_a_rejection_under_a_different_check" in result.stdout
    assert "test_causes_the_rejection_it_claims" not in result.stdout


@pytest.mark.slow
def test_a_marker_with_no_argument_is_refused_rather_than_ignored(tmp_path: Path):
    """Reported reviewing PR #121, round 8.

    `mark.args` is empty for a bare `@pytest.mark.rejects`, so there was nothing
    to check and the test passed carrying a claim that could never be tested.
    That is the silent-claim shape this whole mechanism exists to remove,
    reappearing inside the mechanism — a marker is only worth having if writing
    it wrong is louder than not writing it.
    """
    result = _run_pytest(
        tmp_path,
        """
        import pytest

        @pytest.mark.rejects
        def test_bare_marker():
            assert True

        @pytest.mark.rejects()
        def test_empty_parens():
            assert True

        @pytest.mark.rejects("")
        def test_empty_string():
            assert True
        """,
    )

    # Asserted as behaviour, not as output shape. `rejects("")` also fails as an
    # *unearned* marker — no verdict satisfies an empty claim — so "the phrase
    # appears" passed with the empty-string case unhandled. Counting the phrase
    # instead gave 3 locally and 6 on CI, and pinning the width still left
    # pytest truncating the reason. What actually distinguishes the two is that
    # none of the three may be reported as unearned.
    assert len(_failures(result)) == 3, result.stdout
    assert "unearned" not in result.stdout, result.stdout


@pytest.mark.slow
def test_a_rejection_caused_by_a_fixture_does_not_earn_the_marker(tmp_path: Path):
    """Reported reviewing PR #121, round 8.

    The recorder starts before the test's other fixtures do, so a rejection
    produced during *setup* satisfied the claim and the body was never required
    to cause anything. The marker says this test proves a gate can say no; a
    fixture saying it does not make that true.

    Held to the call phase, both directions, because a fix that simply stopped
    recording setup would break every test whose capability is driven through a
    fixture — the second test here is that case and must still pass.
    """
    result = _run_pytest(
        tmp_path,
        """
        import pytest
        from labpilot.research_engine.execution.schemas import TaskEvidence

        def _reject():
            TaskEvidence(task_id="t", execution_id="e", capability="reporting",
                         checks=["reflect"], passed=False)

        @pytest.fixture
        def rejects_during_setup():
            _reject()
            yield

        @pytest.mark.rejects("reporting:reflect")
        def test_lets_its_fixture_do_the_proving(rejects_during_setup):
            assert True

        @pytest.mark.rejects("reporting:reflect")
        def test_causes_it_in_the_body_itself(rejects_during_setup):
            _reject()
        """,
    )

    assert "1 failed" in result.stdout and "1 passed" in result.stdout, result.stdout
    assert "test_lets_its_fixture_do_the_proving" in result.stdout
    assert "test_causes_it_in_the_body_itself" not in result.stdout


@pytest.mark.slow
def test_installing_the_hook_without_the_fixture_fails_the_test_not_the_session(tmp_path: Path):
    """Reported reviewing PR #121, round 8, as an `INTERNALERROR`.

    `item.stash[_OBSERVED]` subscripted a key that the line above read with
    `.get(..., [])`, so an unset stash plus an unmet claim raised `KeyError`
    inside a report hookwrapper — which pytest escalates to `INTERNALERROR`,
    killing the whole session. A defensive read and an unguarded one, two lines
    apart, which is the shape round 7 was about.

    The stash is unset whenever the fixture is not installed, and the hook and
    the fixture are separate imports with nothing coupling them.
    """
    result = _run_pytest(
        tmp_path,
        """
        import pytest

        @pytest.mark.rejects("reporting:reflect")
        def test_marked_but_the_fixture_never_ran():
            assert True
        """,
        conftest="from helpers.verdict_observer import pytest_runtest_makereport  # noqa: F401",
    )

    assert "INTERNALERROR" not in result.stdout + result.stderr, result.stdout + result.stderr
    assert "1 failed" in result.stdout, result.stdout


def test_the_conftest_installs_every_hook_this_module_defines():
    """The root cause of the `INTERNALERROR` above, rather than its symptom.

    The observer is spread over a fixture and two hooks that a conftest has to
    import by name, and any subset of them imports cleanly. A partial install is
    silent in both directions: the fixture without the hooks records and checks
    nothing, the hooks without the fixture check against an empty recording.
    `pytest_plugins` would couple them, but pytest only honours it in the
    rootdir conftest and this suite's lives in `tests/`.
    """
    from helpers import verdict_observer

    installed = (Path(__file__).resolve().parents[1] / "conftest.py").read_text(encoding="utf-8")
    exported = [name for name in vars(verdict_observer) if name.startswith("pytest_")]

    assert exported, "no hooks found — has the module been renamed?"
    missing = [name for name in [*exported, "verdict_observer"] if name not in installed]

    assert not missing, f"tests/conftest.py does not install: {missing}"
