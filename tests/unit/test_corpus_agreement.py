"""What tier 3 compares, and what it deliberately does not.

The integration test runs this against five real datasets and they all agree —
which is the answer you want and the answer that proves nothing about the
comparison itself. These drive `disagreements` with hand-built scorecards, so a
green tier 3 means the criteria matched rather than that nothing was checked.
"""

from __future__ import annotations

from labpilot.accessor.benchmark.score import CriterionResult, Scorecard, disagreements


def _card(**verdicts: str) -> Scorecard:
    return Scorecard(
        slug="demo",
        results=[CriterionResult(criterion=k, verdict=v) for k, v in verdicts.items()],
    )


def test_matching_verdicts_are_not_a_disagreement() -> None:
    both = _card(target_column="pass", modality="pass")

    assert disagreements(both, both) == {}


def test_a_criterion_the_dataset_reads_differently_is_reported() -> None:
    """The case that licenses the whole corpus. A fixture that passes where the
    real dataset fails is a fixture answering a question it cannot see."""
    hermetic = _card(target_column="pass")
    full = _card(target_column="fail")

    assert disagreements(hermetic, full) == {"target_column": ("pass", "fail")}


def test_unverifiable_is_not_held_against_the_fixture() -> None:
    """`unverifiable` is the fixture saying it cannot speak to a criterion.

    Holding it to one would be asserting that a truncation preserved exactly
    what it declared it destroyed — `feature_columns` is `unverifiable` on every
    headers-only capture and resolves fine on real rows.
    """
    hermetic = _card(feature_columns="unverifiable")
    full = _card(feature_columns="pass")

    assert disagreements(hermetic, full) == {}


def test_not_applicable_is_not_held_against_it_either() -> None:
    hermetic = _card(metric_name="not_applicable")
    full = _card(metric_name="pass")

    assert disagreements(hermetic, full) == {}


def test_a_known_failure_that_stops_failing_on_real_data_is_a_disagreement() -> None:
    """A defect the fixture ships red on purpose must be red on the dataset too.

    If it is not, the fixture is reproducing something the real data does not do
    — which is the capture inventing a failure rather than recording one.
    """
    hermetic = _card(metric_name="known_failure")
    full = _card(metric_name="pass")

    assert disagreements(hermetic, full) == {"metric_name": ("known_failure", "pass")}


def test_a_criterion_the_full_run_never_reached_is_reported_as_missing() -> None:
    """Silence is not agreement. A criterion the full scorer did not produce is
    a comparison that did not happen, and reporting it as a match would be the
    corpus licensing itself."""
    hermetic = _card(abstention="pass")
    full = _card(target_column="pass")

    assert disagreements(hermetic, full) == {"abstention": ("pass", "missing")}
