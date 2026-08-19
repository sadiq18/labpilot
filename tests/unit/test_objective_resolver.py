"""An objective the system cannot justify does not get experiments run against it.

Everything here is measured from this repository, not hypothesised:

* `playground-series-s6e7/competition.json` states `balanced_accuracy_score` and
  every campaign scored plain `accuracy`, because nothing implements it and the
  selector quietly fell back.
* `rogii/competition.json` has `evaluation_metric: None` — a workspace that ran
  campaigns for two weeks with no stated objective at all.
* All fifteen of rogii's evidence cards were built as though MSE were maximised,
  which is the contradiction case below.
"""

from __future__ import annotations

import pytest

from labpilot.research_engine.intelligence.competition.objective import (
    ACTIONABLE_CONFIDENCE,
    infer_direction_from_name,
    resolve_objective,
)


def _resolve(raw, **kw):
    kw.setdefault("task", "tabular_regression")
    return resolve_objective(metric_raw=raw, **kw)


# --- the cases that cost real campaigns -------------------------------------


def test_a_declaration_that_contradicts_the_evaluator_stops_everything() -> None:
    """The rogii failure. Fifteen cards were built as though MSE were maximised;
    the single genuine improvement was recorded `rejected`.

    Neither source wins — one of them is wrong, and picking is how the sign got
    inverted in the first place."""
    objective = _resolve("rmse", declared_direction="maximize")

    assert objective.blocks_launch
    assert objective.contradiction is not None
    assert "minimize" in objective.contradiction and "maximize" in objective.contradiction
    assert objective.direction is None
    assert objective.confidence == 0.0


def test_the_competitions_own_metric_string_resolves() -> None:
    """`mean_squared_error` matched nothing, so an `mse` target could never fire."""
    objective = _resolve("mean_squared_error")

    assert objective.metric_name == "mse"
    assert objective.direction == "minimize"
    assert objective.is_actionable


def test_a_metric_nothing_can_compute_does_not_launch() -> None:
    """The live mis-map, surfaced instead of substituted. Knowing the objective
    and being able to measure it are different capabilities."""
    objective = resolve_objective(
        metric_raw="balanced_accuracy_score", task="tabular_classification"
    )

    assert objective.metric_name == "balanced_accuracy"   # named correctly
    assert objective.direction == "maximize"              # and oriented
    assert objective.scorable is False                    # but not computable
    assert objective.blocks_launch
    assert "optimise a proxy" in objective.why_blocked()


def test_a_competition_with_no_stated_metric_blocks() -> None:
    """rogii, exactly: `evaluation_metric: None`, campaigns run anyway."""
    objective = _resolve(None)

    assert objective.blocks_launch
    assert objective.unresolved == ["metric"]
    assert "no evaluation metric" in objective.why_blocked()


# --- the scaling property ---------------------------------------------------


def test_an_uncatalogued_metric_still_gets_a_stable_identity() -> None:
    """A slug is enough to compare two readings of the same quantity. Only
    *aliasing* needs a catalogue, which is why the table does not have to grow
    with the objective space."""
    objective = _resolve("Weighted RMSSE")

    assert objective.metric_name == "weighted_rmsse"
    assert objective.metric_raw == "Weighted RMSSE"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("wellbore_misfit_error", "minimize"),
        ("reconstruction_loss", "minimize"),
        ("mean_regret", "minimize"),
        ("ndcg_at_10", "maximize"),
        ("player_win_rate", "maximize"),
        ("dice_coefficient", "maximize"),
    ],
)
def test_morphology_orients_a_metric_no_registry_knows(name, expected) -> None:
    direction, hint = infer_direction_from_name(name)
    assert direction == expected
    assert hint and hint in name


def test_a_name_pulling_both_ways_is_not_resolved_by_list_order() -> None:
    """`error_score` matches both families. Answering would make the result
    depend on which tuple was checked first — position standing in for evidence,
    which is the failure `tabular.py` records five rounds of."""
    assert infer_direction_from_name("mean_absolute_error_score") == (None, None)


def test_morphology_earns_less_than_measurement() -> None:
    """A guess from a name must not read as strongly as an observation."""
    from labpilot.research_engine.intelligence.competition.objective import _CONFIDENCE

    assert _CONFIDENCE["rules"] < _CONFIDENCE["registry"] < _CONFIDENCE["measured"]
    assert _CONFIDENCE["llm"] < ACTIONABLE_CONFIDENCE, (
        "an LLM-proposed objective must always ask before it is used"
    )


# --- measurement outranks declaration ---------------------------------------


def test_direction_comes_from_the_evaluator_when_it_can_be_run() -> None:
    objective = _resolve("rmse")

    assert objective.direction_source == "measured"
    assert any("probed the evaluator" in e for e in objective.evidence)


def test_an_agreeing_declaration_does_not_downgrade_the_measurement() -> None:
    agreed = _resolve("rmse", declared_direction="minimize")

    assert agreed.contradiction is None
    assert agreed.direction == "minimize"
    assert agreed.direction_source == "measured"


# --- the contract a preflight depends on ------------------------------------


@pytest.mark.parametrize(
    "raw", ["rmse", "mean_squared_error", "balanced_accuracy_score", "WRMSSE", None]
)
def test_blocking_always_carries_an_actionable_reason(raw) -> None:
    """A gate that says no without saying why is a gate nobody can clear."""
    objective = _resolve(raw)

    if objective.blocks_launch:
        assert objective.why_blocked(), f"{raw!r} blocked with no reason"
    else:
        assert objective.why_blocked() == ""


def test_an_actionable_objective_has_every_part_it_needs() -> None:
    objective = _resolve("Root Mean Squared Error")

    assert objective.is_actionable
    assert objective.metric_name and objective.direction and objective.scorable
    assert not objective.unresolved and objective.contradiction is None
    assert objective.confidence >= ACTIONABLE_CONFIDENCE


def test_confidence_is_the_weakest_link_not_the_strongest() -> None:
    """The identity came from the registry and the direction from measurement.
    An objective is only as good as its weakest part, so nine strong signals
    cannot average away one weak one."""
    objective = _resolve("rmse")

    assert objective.direction_source == "measured"
    assert objective.confidence == pytest.approx(0.90)  # registry, not 0.99


def test_an_unorientable_metric_is_not_guessed() -> None:
    """Direct cover for the branch every other test masked.

    A metric with no probe, no declaration, no registry entry and no
    morphological hint has no direction — and `unresolved` must say so. Every
    uncatalogued metric is *also* unscorable, so the scoring block hid this one:
    a mutation defaulting the direction to "maximize" kept all twenty-two tests
    green because those objectives still blocked, for the other reason.
    """
    objective = _resolve("WRMSSE")

    assert objective.direction is None
    assert "direction" in objective.unresolved
    assert objective.direction_source == "unknown"


def test_resolve_direction_returns_nothing_when_nothing_says_anything() -> None:
    from labpilot.research_engine.intelligence.competition.objective import resolve_direction

    direction, source, _evidence, contradiction = resolve_direction("wrmsse")

    assert direction is None
    assert source == "unknown"
    assert contradiction is None


# --- review findings --------------------------------------------------------


def test_source_always_explains_the_confidence() -> None:
    """Review finding. `source` reported the *stronger* of the two inputs while
    `confidence` was the `min`, so `source='measured', confidence=0.90` gave no
    way to see that the registry had capped it."""
    objective = _resolve("rmse")

    assert objective.identity_source == "registry"
    assert objective.direction_source == "measured"
    assert objective.source == "registry", "source must name the input that capped confidence"
    assert objective.confidence == pytest.approx(_confidence_of(objective.source))


def _confidence_of(source: str) -> float:
    from labpilot.research_engine.intelligence.competition.objective import _CONFIDENCE

    return _CONFIDENCE[source]


@pytest.mark.parametrize(
    ("raw", "declared"),
    [("WRMSSE", None), ("rmse", "maximize")],
)
def test_a_blocked_objective_offers_the_answers_it_could_take(raw, declared) -> None:
    """Review finding. `alternatives` was documented as what lets a question
    offer real choices and was never populated, so the interactive prompt was a
    bare yes/no."""
    objective = _resolve(raw, declared_direction=declared)

    assert objective.blocks_launch
    assert objective.alternatives == ["maximize", "minimize"]


def test_a_resolved_objective_offers_nothing_to_choose() -> None:
    """Empty means there is nothing to decide, not that it went unrecorded."""
    assert _resolve("rmse").alternatives == []


def test_there_is_one_slug_implementation() -> None:
    """Review finding. `_slug_identity` restated `metric_vocabulary._slug`, and
    the identity of every uncatalogued metric depended on the two regexes never
    diverging."""
    from labpilot.research_engine.intelligence.competition import objective as objective_module

    assert not hasattr(objective_module, "_slug_identity")
    assert _resolve("Weighted RMSSE").metric_name == "weighted_rmsse"
