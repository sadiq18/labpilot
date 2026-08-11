"""`rank_candidates` must pick one winner, always the same one (M11).

The cohort verdict is recomputed from scratch every time a branch lands, so a
ranking that depends on argument order would let the same K results promote
different branches on different arrivals.
"""

from __future__ import annotations

from labpilot.research_engine.agents.promotion import rank_candidates


def _c(name: str, value=None, completed_at: str | None = None, **metric_fields) -> dict:
    """A candidate as `_candidates_for` builds one.

    `metric_fields` exists so a test can produce the metrics dict a *real*
    run writes, not only the tidy `{key: float}` shape — a placeholder run's
    `status` marker is a field of `metrics`, and a fixture that cannot express
    it cannot catch a ranker that ignores it.
    """
    metrics = {} if value is None else {"cv_rmse": value}
    metrics.update(metric_fields)
    return {"experiment_id": name, "metrics": metrics, "completed_at": completed_at}


def _stub(name: str, value: float) -> dict:
    """Exactly what `ExperimentSpecialist` records for a dry-run branch."""
    return _c(name, value, status="dry_run_stub")


def test_lowest_wins_when_minimising() -> None:
    best = rank_candidates(
        [_c("a", 0.5), _c("b", 0.2), _c("c", 0.9)], "cv_rmse", maximize=False
    )
    assert best["experiment_id"] == "b"


def test_highest_wins_when_maximising() -> None:
    best = rank_candidates(
        [_c("a", 0.5), _c("b", 0.2), _c("c", 0.9)], "cv_rmse", maximize=True
    )
    assert best["experiment_id"] == "c"


def test_a_tie_goes_to_the_branch_that_finished_first() -> None:
    best = rank_candidates(
        [
            _c("late", 0.5, "2026-08-11T10:00:05+00:00"),
            _c("early", 0.5, "2026-08-11T10:00:01+00:00"),
        ],
        "cv_rmse",
        maximize=False,
    )
    assert best["experiment_id"] == "early"


def test_a_tie_is_broken_the_same_way_whatever_the_argument_order() -> None:
    a = _c("a", 0.5, "2026-08-11T10:00:01+00:00")
    b = _c("b", 0.5, "2026-08-11T10:00:05+00:00")

    assert rank_candidates([a, b], "cv_rmse", maximize=False)["experiment_id"] == "a"
    assert rank_candidates([b, a], "cv_rmse", maximize=False)["experiment_id"] == "a"


def test_a_tie_with_no_timestamps_still_settles_deterministically() -> None:
    a, b = _c("a", 0.5), _c("b", 0.5)

    assert rank_candidates([a, b], "cv_rmse", maximize=False)["experiment_id"] == "a"
    assert rank_candidates([b, a], "cv_rmse", maximize=False)["experiment_id"] == "a"


def test_a_member_without_a_timestamp_loses_the_tie() -> None:
    """Absence must not win by sorting first."""
    best = rank_candidates(
        [_c("unstamped", 0.5), _c("stamped", 0.5, "2026-08-11T10:00:09+00:00")],
        "cv_rmse",
        maximize=False,
    )
    assert best["experiment_id"] == "stamped"


def test_a_diverged_run_does_not_win_by_default() -> None:
    """Every NaN comparison is False, so an unfiltered NaN wins under `min`."""
    best = rank_candidates(
        [_c("nan", float("nan")), _c("real", 0.9)], "cv_rmse", maximize=False
    )
    assert best["experiment_id"] == "real"


def test_an_infinite_score_is_not_comparable_either() -> None:
    best = rank_candidates(
        [_c("inf", float("-inf")), _c("real", 0.9)], "cv_rmse", maximize=False
    )
    assert best["experiment_id"] == "real"


def test_a_boolean_metric_is_not_a_score() -> None:
    """`True` is an `int` in Python and would otherwise rank as 1.0."""
    best = rank_candidates(
        [_c("flag", True), _c("real", 5.0)], "cv_rmse", maximize=True
    )
    assert best["experiment_id"] == "real"


def test_candidates_missing_the_metric_are_skipped() -> None:
    best = rank_candidates(
        [_c("none"), {"experiment_id": "no-metrics"}, _c("real", 0.4)],
        "cv_rmse",
        maximize=False,
    )
    assert best["experiment_id"] == "real"


def test_a_run_that_never_trained_a_model_cannot_win() -> None:
    """The rogii 2026-08-07 failure, as a cohort: a stub's 0.5 against a real
    run's 194.80. Nothing about a placeholder's number is comparable, and on
    an error metric it beats every genuine result.
    """
    best = rank_candidates(
        [_stub("stub", 0.5), _c("real", 194.80)], "cv_rmse", maximize=False
    )
    assert best["experiment_id"] == "real"


def test_the_other_placeholder_marker_is_refused_too() -> None:
    """`PLACEHOLDER_STATUSES` holds two; a check that hardcoded one would pass
    the test above and still promote a scaffold."""
    best = rank_candidates(
        [_c("scaffold", 0.5, status="last_resort_scaffold"), _c("real", 194.80)],
        "cv_rmse",
        maximize=False,
    )
    assert best["experiment_id"] == "real"


def test_an_all_placeholder_cohort_has_no_winner() -> None:
    assert rank_candidates([_stub("a", 0.5), _stub("b", 0.1)], "cv_rmse", maximize=False) is None


def test_nothing_comparable_means_no_winner() -> None:
    assert rank_candidates([_c("a"), _c("b")], "cv_rmse", maximize=False) is None
    assert rank_candidates([], "cv_rmse", maximize=False) is None
