"""The observer, held to the standard it exists to enforce.

This mechanism decides whether every other rejection test in the suite counts,
so the question it has to answer first is its own: *can it fail?* Each test
below is written from the failure it is meant to catch — a marker on a test that
rejects nothing, a marker naming the wrong gate, a recorder that keeps recording
after its block — rather than from the behaviour it is meant to allow.
"""

from __future__ import annotations

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
    tests = Path(__file__).resolve().parents[1]
    (tmp_path / "conftest.py").write_text(
        textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(tests)!r})
        from helpers.verdict_observer import (  # noqa: F401
            pytest_runtest_makereport,
            verdict_observer,
        )

        def pytest_configure(config):
            config.addinivalue_line("markers", "rejects: M20")
        """),
        encoding="utf-8",
    )
    (tmp_path / "test_lying.py").write_text(_LYING_TEST, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert "1 passed" in result.stdout and "2 failed" in result.stdout, result.stdout
    assert "test_claims_a_rejection_it_never_causes" in result.stdout
    assert "test_causes_a_rejection_under_a_different_check" in result.stdout
    assert "test_causes_the_rejection_it_claims" not in result.stdout
