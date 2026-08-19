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


# --- the control half of the seam -------------------------------------------


@pytest.mark.parametrize("control_metrics", [None, {}])
def test_a_control_that_scored_without_a_blob_is_still_a_control(
    tmp_path, control_metrics
) -> None:
    """Review finding, and the one that blocked phase 2.

    Both the assignment and the extraction asked the metrics *blob* rather than
    the result, so a validator that scores without writing Kaggle-shaped `cv_`
    keys had its control silently discarded and every comparison came back
    `missing_control`. Parametrized over both empty forms because `is None` and
    falsy are different tests and only one of them was wrong at each site —
    `resolve_control` returns `{}`, which passed the `is None` check.

    The treatment side never had the bug: `_found(result, ...)` is unguarded.
    That asymmetry is why 3088 green tests said nothing.
    """
    kwargs = {} if control_metrics is None else {"control_metrics": control_metrics}
    card = _card(
        tmp_path,
        "demo-blobless",
        treatment_metrics={},
        result=ValidationResult(
            score=0.90, metric="pass_rate", direction="maximize", source="harness"
        ),
        control_result=ValidationResult(
            score=0.70, metric="pass_rate", direction="maximize", source="harness"
        ),
        **kwargs,
    )

    assert card.observed.parent_cv == 0.70, "the control result was thrown away"
    assert card.observed.cv_gain == pytest.approx(0.20)
    assert card.decision.value == "accepted"


def test_a_control_result_with_no_score_is_still_a_missing_control(tmp_path) -> None:
    """The other side of it. Carrying a result is not the same as having scored,
    and treating one as a control would compare against nothing."""
    card = _card(
        tmp_path,
        "demo-scoreless-control",
        treatment_metrics={},
        result=ValidationResult(
            score=0.90, metric="pass_rate", direction="maximize", source="harness"
        ),
        control_result=ValidationResult(
            score=None, metric="", direction="maximize", source="harness"
        ),
    )

    assert card.observed.parent_cv is None
    assert card.decision_reason == "missing_control"


# --- the validator actually runs --------------------------------------------


def _workspace(tmp_path: Path, *, direction: str) -> Path:
    import json

    (tmp_path / "metrics.json").write_text(
        json.dumps({"cv_rmse": 1.5, "cv_std": 0.1}), encoding="utf-8"
    )
    (tmp_path / "competition.json").write_text(
        json.dumps(
            {
                "slug": "demo",
                "evaluation_metric": {"name": "rmse", "key": "rmse", "direction": direction},
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


class _Context:
    def __init__(self, root: Path) -> None:
        self.competition = "demo"
        self.workspace_root = root
        self.paths = None


def test_the_kaggle_validator_reads_the_workspace_it_was_given(tmp_path) -> None:
    """Review finding. `getattr(workspace, "root", workspace)` reads as "use
    `.root` if this is a Workspace, else treat it as a path" and is not:
    `pathlib.Path` *has* a `root` property returning "/". Passing the obvious
    thing — a `Path`, which is what `TaskContext.workspace_root` holds — made the
    validator read `/metrics.json` and return a scoreless result with no error.

    Nothing caught it because the only test of this method asserted that it was
    callable.
    """
    root = _workspace(tmp_path, direction="minimize")

    result = KaggleCvValidator().validate(None, root, _Context(root))

    assert result.score == 1.5
    assert result.metric == "cv_rmse"
    assert result.direction == "minimize"
    assert result.artifacts["metrics"] == str(root / "metrics.json")


def test_a_workspace_object_works_too(tmp_path) -> None:
    """The shape the `getattr` was actually written for. Both must work, or
    fixing one caller breaks the other."""
    root = _workspace(tmp_path, direction="maximize")

    class _Workspace:
        def __init__(self, path: Path) -> None:
            self.root = path

    class _NoRootContext:
        competition = "demo"
        paths = None

    result = KaggleCvValidator().validate(None, _Workspace(root), _NoRootContext())

    assert result.score == 1.5
    assert result.direction == "maximize"


def test_a_workspace_with_no_metrics_reports_no_score(tmp_path) -> None:
    """An absent metrics.json is "this run scored nothing", not a crash — and it
    must be distinguishable from the `/metrics.json` misread it used to produce,
    which looked exactly the same."""
    (tmp_path / "competition.json").write_text(
        '{"slug": "demo", "evaluation_metric": {"name": "rmse", "key": "rmse", '
        '"direction": "minimize"}}',
        encoding="utf-8",
    )

    result = KaggleCvValidator().validate(None, tmp_path, _Context(tmp_path))

    assert result.score is None
    assert result.direction == "minimize", "direction is independent of the score"
    assert not result.is_comparable


def test_a_control_result_supplies_the_blob_the_card_reads_around_it(tmp_path) -> None:
    """The score is not the only thing a card takes from the control side.

    `cv_std`, `train_time_s` and `peak_memory_mb` are read straight off the
    control blob, and stability is derived from the two spreads. So a
    `control_result` must hand over its `raw`, not just its score — otherwise
    stability silently reads `UNKNOWN` for every validator that supplies one.

    Mutation finding: with an empty `raw` on both sides the assignment changes
    nothing, so restoring the old `is None` test kept the suite green.
    """
    card = _card(
        tmp_path,
        "demo-control-blob",
        treatment_metrics={},
        result=ValidationResult(
            score=190.97,
            metric="cv_rmse",
            direction="minimize",
            source="harness",
            raw={"cv_rmse": 190.97, "cv_std": 1.1, "train_time_s": 100.0},
        ),
        control_metrics={},
        control_result=ValidationResult(
            score=194.80,
            metric="cv_rmse",
            direction="minimize",
            source="harness",
            raw={"cv_rmse": 194.80, "cv_std": 1.1, "train_time_s": 80.0},
        ),
    )

    assert card.observed.parent_cv_std == 1.1, "the control blob never arrived"
    assert card.observed.stability.value == "similar"
    assert card.decision.value == "accepted"


# --- phase 1: the production path goes through the seam ---------------------


def test_the_validator_consults_every_direction_source_the_builder_does(tmp_path) -> None:
    """The rogii case, and the one an earlier draft of this validator failed.

    `_resolve_direction` asks `resolve_maximize` with `ResearchPaths.root` *and*
    `ResearchPaths.extracted_dir`. The first version of `KaggleCvValidator`
    passed `paths.base_dir` as the knowledge root and omitted the extracted
    directory, so it consulted strictly fewer sources than the builder it stands
    in for.

    That gap is not small: the Analyze profile artifact under `extracted_dir` is
    where rogii's `minimize` actually lived. Routing production through a
    validator that answers `None` here would have made the campaign refuse to
    build a card it builds today — a regression dressed as a refactor.
    """
    import json

    from labpilot.research_engine.intelligence.paths import ResearchPaths
    from labpilot.research_engine.validation.kaggle import direction_for

    competition = "rogii-shaped"
    knowledge = tmp_path / "knowledge"
    paths = ResearchPaths(knowledge, competition)
    misc = paths.extracted_dir / "misc"
    misc.mkdir(parents=True, exist_ok=True)
    # Direction lives *only* here — no competition.json anywhere.
    (misc / f"competition_{competition}.json").write_text(
        json.dumps({"metadata": {"profile": {"metric": {"name": "mse", "direction": "minimize"}}}}),
        encoding="utf-8",
    )

    workspace = tmp_path / "ws"
    workspace.mkdir()

    assert direction_for(competition, knowledge_dir=knowledge, workspace_root=workspace) is False


def test_the_validator_matches_what_the_builder_would_have_resolved(tmp_path) -> None:
    """The no-op, at the direction boundary specifically. Two implementations of
    "which way is better" that disagree would move every baseline."""
    import json

    from labpilot.research_engine.evidence.builder import _resolve_direction
    from labpilot.research_engine.validation.kaggle import direction_for

    competition = "demo-agree"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "competition.json").write_text(
        json.dumps(
            {
                "slug": competition,
                "evaluation_metric": {"name": "rmse", "key": "rmse", "direction": "minimize"},
            }
        ),
        encoding="utf-8",
    )

    assert direction_for(
        competition, knowledge_dir=tmp_path, workspace_root=workspace
    ) == _resolve_direction(tmp_path, competition, workspace)


def test_an_unresolvable_direction_is_carried_not_raised(tmp_path) -> None:
    """The validator reports `None`; `build_evidence_card` is the layer that
    decides refusing is the right response. Splitting it the other way would put
    the refusal in a place with no idea what the caller can substitute."""
    from labpilot.research_engine.validation.kaggle import direction_for

    workspace = tmp_path / "ws"
    workspace.mkdir()

    assert direction_for("nothing-stated", knowledge_dir=tmp_path, workspace_root=workspace) is None


def test_the_production_comparison_runs_through_the_validator() -> None:
    """Phase 1's whole content. Until this, `validate` was available and unused,
    so the seam could rot without a single test noticing."""
    import inspect

    from labpilot.research_engine.evidence import compare_service

    source = inspect.getsource(compare_service.run_compare_and_build_card)

    assert "KaggleCvValidator().validate(" in source
    assert "result=result" in source and "control_result=control_result" in source
    assert "_load_metrics(root" not in source, "still loading metrics.json around the validator"


def test_the_knowledge_root_is_the_research_dir_not_the_knowledge_dir(tmp_path) -> None:
    """Mutation finding. `ResearchPaths.root` is `<base_dir>/<competition>/research`,
    not `base_dir` — so passing `base_dir` straight through as `knowledge_root`
    looks equivalent and searches a directory that holds no contract.

    Every other test placed the contract where both spellings happen to find it
    (or nowhere), so the two were indistinguishable. Here the knowledge copy of
    `competition.json` sits only where `_resolve_direction` actually looks.
    """
    import json

    from labpilot.research_engine.intelligence.paths import ResearchPaths
    from labpilot.research_engine.validation.kaggle import direction_for

    competition = "demo-knowledge-root"
    paths = ResearchPaths(tmp_path, competition)
    paths.root.mkdir(parents=True, exist_ok=True)
    (paths.root / "competition.json").write_text(
        json.dumps({"metric": {"name": "rmse", "direction": "minimize"}}), encoding="utf-8"
    )

    workspace = tmp_path / "ws"
    workspace.mkdir()

    assert direction_for(competition, knowledge_dir=tmp_path, workspace_root=workspace) is False
