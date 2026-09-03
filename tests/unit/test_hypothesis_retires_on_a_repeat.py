"""A hypothesis is retired for repeating a failure, not for failing three times.

Issue #176, the sibling of #173 one layer down. `DEFAULT_MAX_ATTEMPTS`'s own
comment already said the assumption out loud — *"the fourth **identical**
failure teaches nothing the third did not"* — and `classify_hypothesis_failure`
never asked whether the failures were identical.

It matters more here than it did for the campaign breaker. `stop:failing` ends a
run an operator can restart with a bigger budget; this writes `REJECTED` onto the
hypothesis, and `load_open_hypothesis_tags` then treats the technique as covered,
so the campaign will not propose it again. A converging repair loop that runs out
of attempts does not just lose the run — it teaches the system the technique
failed, which is the shape of the false `vit` claims M14's repair chain exists to
undo.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from labpilot.research_engine.execution.engineer import ResearchEngineer, default_stub_registry
from labpilot.research_engine.execution.store import ExecutionStore
from labpilot.research_engine.planner.schemas.models import ResearchPlan, ResearchTask
from labpilot.research_engine.planner.schemas.task_types import (
    PlanStatus,
    TaskType,
)
from labpilot.research_engine.planner.store import PlanStore
from labpilot.research_engine.reflection.hypotheses.outcomes import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_DISTINCT_ATTEMPTS,
    HypothesisOutcome,
    classify_hypothesis_failure,
)

#: The three failures from the campaign in #173, in the order they surfaced.
CONVERGING = (
    "pipeline/infer.py imports train but its PEP 723 block does not declare it",
    "LGBMClassifier.fit() got an unexpected keyword argument 'verbose'",
    "pandas comparison error in arraylike.py",
)
STUCK = "ModuleNotFoundError: No module named 'catboost'"


def test_three_distinct_failures_do_not_retire_the_hypothesis() -> None:
    """The issue's case: each attempt fixed the last defect and found a new one."""
    outcome, why = classify_hypothesis_failure(
        failure_reason=CONVERGING[-1],
        attempts=DEFAULT_MAX_ATTEMPTS,
        recent_failures=CONVERGING,
    )

    assert outcome is HypothesisOutcome.RETRYABLE
    assert "not yet evidence about the hypothesis" in why


def test_three_identical_failures_still_retire_it() -> None:
    """The behaviour the default was chosen for."""
    outcome, why = classify_hypothesis_failure(
        failure_reason=STUCK,
        failure_kind="other",
        attempts=DEFAULT_MAX_ATTEMPTS,
        recent_failures=(STUCK, STUCK, STUCK),
    )

    assert outcome is HypothesisOutcome.DEAD_END
    assert "with the same failure" in why


def test_an_oscillating_loop_is_a_repeat() -> None:
    """Fixing A reintroduces B and back. Every failure differs from the one
    before it and nothing is converging."""
    a, b = CONVERGING[0], CONVERGING[1]
    outcome, _ = classify_hypothesis_failure(
        attempts=DEFAULT_MAX_ATTEMPTS, recent_failures=(a, b, a)
    )

    assert outcome is HypothesisOutcome.DEAD_END


def test_endlessly_novel_failures_still_hit_a_ceiling() -> None:
    """The backstop. The campaign breaker could hand its distinct case to
    `max_barren_steps`; there is no equivalent here, so without a ceiling a
    hypothesis whose failures never repeat would never retire and the selector
    would keep offering it."""
    novel = tuple(f"{word} exploded" for word in "alpha bravo charlie delta echo".split())

    below, _ = classify_hypothesis_failure(
        attempts=DEFAULT_MAX_DISTINCT_ATTEMPTS - 1, recent_failures=novel
    )
    at, why = classify_hypothesis_failure(
        attempts=DEFAULT_MAX_DISTINCT_ATTEMPTS, recent_failures=novel
    )

    assert below is HypothesisOutcome.RETRYABLE
    assert at is HypothesisOutcome.DEAD_END
    # The ceiling's message does not claim the failures were the same.
    assert "with the same failure" not in why


def test_the_ceiling_is_reachable_within_a_campaign() -> None:
    """Both bounds, and the upper one is the point.

    The test above phrases the ceiling in terms of the constant, so it moves
    with it and cannot fail: raising `DEFAULT_MAX_DISTINCT_ATTEMPTS` to 10,000
    passed every other test here. A ceiling a campaign can never reach is not a
    ceiling — the campaign breakers stop a run long before that many executions
    — so the number is pinned to a range a real run can cross.
    """
    assert DEFAULT_MAX_DISTINCT_ATTEMPTS > DEFAULT_MAX_ATTEMPTS, "the repeat rule is unreachable"
    assert DEFAULT_MAX_DISTINCT_ATTEMPTS <= 20, (
        "a ceiling no campaign reaches leaves an endlessly-novel hypothesis "
        "retiring only when the run ends"
    )


def test_a_hypothesis_failing_ten_novel_ways_is_retired() -> None:
    """Stated in a literal count rather than the constant, so the backstop is
    asserted against a number rather than against itself."""
    novel = tuple(f"{word} exploded" for word in "alpha bravo charlie delta echo".split())

    outcome, _ = classify_hypothesis_failure(attempts=10, recent_failures=novel)

    assert outcome is HypothesisOutcome.DEAD_END


def test_a_caller_that_supplies_no_history_keeps_the_old_behaviour() -> None:
    """`recent_failures` defaults to empty, and every existing caller and test
    relies on that meaning "retire at the threshold" rather than "never"."""
    outcome, _ = classify_hypothesis_failure(attempts=DEFAULT_MAX_ATTEMPTS)

    assert outcome is HypothesisOutcome.DEAD_END


def test_redundancy_still_settles_it_before_anything_else() -> None:
    """A change already in the parent will be there on every future attempt, so
    no amount of novel failure makes retrying worthwhile."""
    outcome, why = classify_hypothesis_failure(
        attempts=1, recent_failures=CONVERGING, redundant=True
    )

    assert outcome is HypothesisOutcome.DEAD_END
    assert "already implements this change" in why


# --- the plumbing, not the two ends of it ------------------------------------


def _seed_failed_executions(knowledge: Path, errors: list[str]) -> None:
    """A hypothesis with one plan and a failed execution per error."""
    now = datetime.now(UTC)
    plans = PlanStore(knowledge, "demo")
    try:
        plans.upsert_plan(
            ResearchPlan(
                id="P-001",
                competition="demo",
                hypothesis_id="H-001",
                goal="mini",
                status=PlanStatus.READY,
                tasks=[
                    ResearchTask(
                        id="P-001-T01",
                        plan_id="P-001",
                        type=TaskType.PREPARE_WORKSPACE,
                        description="a",
                        order=0,
                    )
                ],
                created_at=now,
                updated_at=now,
            )
        )
    finally:
        plans.close()

    executions = ExecutionStore(knowledge, "demo")
    try:
        for error in errors:
            execution = executions.create_execution("P-001")
            executions.update_status(execution.id, "failed", error=error)
    finally:
        executions.close()


def test_the_engineer_carries_the_failure_texts_not_just_the_count(tmp_path: Path) -> None:
    """The wiring, and the reason this issue was smaller than it looked.

    `_failed_attempts_for` already listed the failed executions to count them,
    and `ResearchExecution.error` is on those very rows — the history was being
    read and discarded. Testing `classify_hypothesis_failure` alone would pass
    whether or not anything ever supplies it, which is the dead-parameter shape
    from #170.
    """
    knowledge = tmp_path / "knowledge"
    _seed_failed_executions(knowledge, list(CONVERGING))
    engineer = ResearchEngineer(
        knowledge_dir=knowledge, competition="demo", registry=default_stub_registry()
    )
    try:
        attempts, errors = engineer._failed_attempts_for("H-001")
    finally:
        engineer.close()

    assert attempts == 3
    assert errors == list(CONVERGING)
    # And what the retirement rule makes of them.
    outcome, _ = classify_hypothesis_failure(attempts=attempts, recent_failures=errors)
    assert outcome is HypothesisOutcome.RETRYABLE


def test_an_unreadable_store_still_reports_one_attempt(tmp_path: Path) -> None:
    """The existing contract: a store that cannot be read means "first attempt",
    never a crash on the failure path."""
    engineer = ResearchEngineer(
        knowledge_dir=tmp_path / "knowledge",
        competition="demo",
        registry=default_stub_registry(),
    )
    try:
        assert engineer._failed_attempts_for("H-404") == (1, [])
    finally:
        engineer.close()


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_executions_with_no_error_text_are_left_out(tmp_path: Path, blank) -> None:
    """They cannot be compared, so carrying them would dilute the history with
    entries that say nothing."""
    knowledge = tmp_path / "knowledge"
    _seed_failed_executions(knowledge, [CONVERGING[0], blank, CONVERGING[1]])
    engineer = ResearchEngineer(
        knowledge_dir=knowledge, competition="demo", registry=default_stub_registry()
    )
    try:
        attempts, errors = engineer._failed_attempts_for("H-001")
    finally:
        engineer.close()

    assert attempts == 3, "a blank failure is still an attempt"
    assert errors == [CONVERGING[0], CONVERGING[1]]
