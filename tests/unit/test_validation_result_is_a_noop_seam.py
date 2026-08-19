"""M12 phase 0: naming the triple must change nothing.

`build_evidence_card` has always taken score, control and direction as loose
arguments assembled from three different Kaggle sources. A `ValidationResult`
is those same three facts as one object that states its own direction — which
is what lets a benchmark harness, with no `competition.json` to write a
direction into, take part at all.

Phase 0 ships only the name. The test that licenses every later phase is that a
card built through the result is identical, field for field, to one built the
old way — because if the wrapper reads different sources, every comparison
afterwards is against a moved baseline and nothing downstream can be trusted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from labpilot.research_engine.evidence.builder import build_evidence_card
from labpilot.research_engine.validation.kaggle import KaggleCvValidator, result_from_metrics
from labpilot.research_engine.validation.models import HypothesisValidator, ValidationResult

#: rogii's one genuine improvement: MSE 194.80 -> 190.97, recorded `rejected`
#: because `maximize` defaulted to True. Fold spread is held equal so the
#: stability branch of `_decide` does not mask the direction under test.
TREATMENT = {"cv_rmse": 190.97, "cv_std": 1.1, "train_time_s": 100.0, "peak_memory_mb": 512.0}
CONTROL = {"cv_rmse": 194.80, "cv_std": 1.1, "train_time_s": 80.0, "peak_memory_mb": 500.0}


def _card(tmp_path: Path, competition: str, **kwargs):
    return build_evidence_card(
        knowledge_dir=tmp_path,
        competition=competition,
        treatment_execution_id="E-014",
        control_execution_id="E-008",
        plan_id="P-002",
        persist=False,
        **kwargs,
    )


def _comparable(card) -> dict:
    """Everything but the identity, which is minted per call."""
    payload = card.model_dump()
    payload.pop("id", None)
    payload.pop("created_at", None)
    return payload


# --- the phase-0 contract ---------------------------------------------------


@pytest.mark.parametrize("maximize", [False, True])
def test_a_card_built_from_a_result_is_identical_to_the_old_way(tmp_path, maximize) -> None:
    """The whole of phase 0. Both directions, because the sign is what this
    milestone is moving and a no-op that only holds for maximize is not one."""
    old = _card(
        tmp_path,
        "demo-old",
        treatment_metrics=TREATMENT,
        control_metrics=CONTROL,
        maximize=maximize,
    )
    new = _card(
        tmp_path,
        "demo-old",
        treatment_metrics={},
        result=result_from_metrics(TREATMENT, maximize=maximize),
        control_metrics=CONTROL,
        control_result=result_from_metrics(CONTROL, maximize=maximize),
    )

    assert _comparable(new) == _comparable(old)


def test_the_direction_rides_on_the_result(tmp_path) -> None:
    """The point of the object. No `maximize=` argument and no `competition.json`
    anywhere — a validator that measured its own direction can now say so, which
    is what a benchmark harness needs and had no way to express."""
    card = _card(
        tmp_path,
        "demo-direction",
        treatment_metrics={},
        result=result_from_metrics(TREATMENT, maximize=False),
        control_metrics=CONTROL,
    )

    assert card.maximize is False
    assert card.observed.cv_gain == pytest.approx(190.97 - 194.80)
    assert card.decision.value == "accepted", "an RMSE that fell is an improvement"


def test_an_explicit_argument_still_wins_over_the_result(tmp_path) -> None:
    """Phase 0 adds a source; it must not silently take precedence over a caller
    that named the direction outright, or the no-op claim above is vacuous."""
    card = _card(
        tmp_path,
        "demo-precedence",
        treatment_metrics={},
        result=result_from_metrics(TREATMENT, maximize=False),
        control_metrics=CONTROL,
        maximize=True,
    )

    assert card.maximize is True


# --- what the result refuses to do ------------------------------------------


def test_a_result_with_no_direction_does_not_invent_one(tmp_path) -> None:
    """`resolve_maximize` returning None is a real answer. The result carries it
    as None rather than as `True`, which is the default that recorded rogii's one
    genuine improvement as `rejected`."""
    unresolved = result_from_metrics(TREATMENT, maximize=None)

    assert unresolved.direction is None
    assert unresolved.maximize is None
    assert not unresolved.is_comparable

    with pytest.raises(ValueError):
        _card(
            tmp_path,
            "demo-unknown",
            treatment_metrics={},
            result=unresolved,
            control_metrics=CONTROL,
        )


def test_a_score_without_a_direction_is_not_comparable() -> None:
    """Both halves are required. `treatment - control` is computable with either
    one missing, and its sign is a coin flip."""
    assert not ValidationResult(score=1.0, metric="rmse", direction=None, source="t").is_comparable
    assert not ValidationResult(
        score=None, metric="rmse", direction="minimize", source="t"
    ).is_comparable
    assert ValidationResult(
        score=1.0, metric="rmse", direction="minimize", source="t"
    ).is_comparable


def test_a_blob_with_no_primary_metric_reports_no_score() -> None:
    """Not zero, and not an exception — the same thing `_primary_cv_keyed`
    returning None has always meant."""
    empty = result_from_metrics({"n_features": 12}, maximize=False)

    assert empty.score is None and empty.metric == ""
    assert not empty.is_comparable


def test_the_metric_key_travels_with_the_score() -> None:
    """So the mismatched-metric refusal does not have to re-read the blob to
    learn what the number was measured on."""
    assert result_from_metrics(TREATMENT, maximize=False).metric == "cv_rmse"
    assert result_from_metrics({"cv_accuracy": 0.9}, maximize=True).metric == "cv_accuracy"


def test_two_runs_scored_on_different_metrics_are_still_refused(tmp_path) -> None:
    """The guard that survives the move. Measured on rogii: six cards recorded a
    gain of -194.30 by subtracting a stub's `cv_accuracy` of 0.5 from a real
    `cv_rmse` of 194.80."""
    card = _card(
        tmp_path,
        "demo-mismatch",
        treatment_metrics={},
        result=result_from_metrics(TREATMENT, maximize=False),
        control_metrics={"cv_accuracy": 0.5},
        control_result=result_from_metrics({"cv_accuracy": 0.5}, maximize=False),
    )

    assert card.observed.cv_gain is None
    assert card.decision_reason.startswith("metric_key_mismatch")
    assert "cv_accuracy" in card.decision_reason and "cv_rmse" in card.decision_reason


# --- the protocol -----------------------------------------------------------


def test_the_kaggle_path_satisfies_the_protocol() -> None:
    """One implementation today. The registry waits for a third, per the plan:
    'One extra validator, hardcoded, will reveal the interface.'"""
    validator: HypothesisValidator = KaggleCvValidator()

    assert validator.source == "kaggle_cv"
    assert callable(validator.validate)


def test_a_result_says_which_validator_produced_it() -> None:
    """Two runs from different validators are not comparable, and `source` is
    how a later phase will be able to tell."""
    assert result_from_metrics(TREATMENT, maximize=False).source == "kaggle_cv"


def test_the_provenance_records_where_both_facts_came_from() -> None:
    """A direction with no account of its origin is what the objective layer
    exists to stop; a result must not reintroduce it one level down."""
    provenance = result_from_metrics(TREATMENT, maximize=False).provenance

    assert any("cv_rmse" in line for line in provenance)
    assert any("minimize" in line for line in provenance)


def test_the_card_reads_the_result_rather_than_re_deriving_it(tmp_path) -> None:
    """Mutation finding. Deleting the result branch entirely kept the suite
    green, because the Kaggle validator extracts with the same function the
    builder falls back to — so both paths agree by construction and every test
    above passes either way.

    The plumbing is only load-bearing if a result whose score disagrees with its
    own blob is honoured. That is exactly the case a non-Kaggle validator is:
    one that computed a number the `cv_`-prefixed search would never find.
    """
    stated = ValidationResult(
        score=5.0,
        metric="pass_rate",
        direction="maximize",
        source="harness",
        raw={"cv_rmse": 190.97},  # what a re-derivation would pick instead
    )
    control = ValidationResult(
        score=4.0, metric="pass_rate", direction="maximize", source="harness", raw={}
    )

    card = _card(
        tmp_path,
        "demo-carried",
        treatment_metrics={},
        result=stated,
        control_metrics={"cv_rmse": 194.80},
        control_result=control,
    )

    assert card.observed.treatment_cv == 5.0, "re-derived cv_rmse instead of the stated score"
    assert card.observed.cv_gain == pytest.approx(1.0)
    assert card.decision.value == "accepted"
