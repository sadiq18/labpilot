"""M12 phase 2: a second validator, and the exit criteria it exists to meet.

The plan's first exit criterion is *"a second validator implemented, with no
change to the Conductor, policy, hypothesis or reflection code"*. A second
implementation that shares the first one's assumptions would confirm the
interface by not exercising it, so `HarnessValidator` was chosen for what it
lacks: no folds, no `cv_` keys, no submission, no leaderboard, and — the point —
**no `competition.json`**, so direction cannot be recovered from a contract even
in principle. It has to come from the thing that computed the score.

Everything here runs hermetically: no Kaggle credentials, no network, no
knowledge store.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from labpilot.research_engine.evidence.builder import build_evidence_card
from labpilot.research_engine.evidence.compare_service import (
    _validator_for,
    run_compare_and_build_card,
)
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import ResearchExecution
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.planner.schemas.models import ResearchPlan, ResearchTask
from labpilot.research_engine.planner.schemas.task_types import PlanStatus, TaskType
from labpilot.research_engine.validation.harness import (
    OBJECTIVE_FILE,
    RESULT_FILE,
    HarnessValidator,
    handles,
    result_from_payload,
)
from labpilot.research_engine.validation.models import HypothesisValidator


def _harness_workspace(root: Path, *, declare: bool = True, **result) -> Path:
    """A workspace with a harness declaration, its result, and nothing Kaggle.

    Two files, because they answer at different times: `harness.json` is what the
    workspace promises and is what the launch preflight reads, `result.json` is
    what a run produced and is what the validator reads.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / RESULT_FILE).write_text(json.dumps(result), encoding="utf-8")
    if declare:
        (root / OBJECTIVE_FILE).write_text(
            json.dumps({k: result[k] for k in ("metric", "direction") if k in result}),
            encoding="utf-8",
        )
    return root


# --- what makes this a second validator rather than a second spelling -------


def test_a_harness_workspace_holds_nothing_kaggle_shaped(tmp_path) -> None:
    """The claim the rest of this file rests on. If a `competition.json` were
    lurking, direction could come from it and the exercise would prove nothing.
    """
    root = _harness_workspace(tmp_path / "ws", score=0.82, metric="pass_rate", direction="maximize")

    assert not (root / "competition.json").exists()
    assert not (root / "metrics.json").exists()
    assert not (root / "sample_submission.csv").exists()


def test_direction_comes_from_the_harness_because_nothing_else_knows(tmp_path) -> None:
    """The inversion this milestone is about. No contract states the direction;
    the thing that computed the score does."""
    root = _harness_workspace(tmp_path / "ws", score=0.82, metric="pass_rate", direction="maximize")

    result = HarnessValidator().validate("H-001", root, object())

    assert result.score == 0.82
    assert result.metric == "pass_rate"
    assert result.direction == "maximize"
    assert result.source == "harness"
    assert any("stated by the harness" in line for line in result.provenance)


def test_a_minimised_harness_objective_is_read_the_other_way(tmp_path) -> None:
    """So the validator is not just returning `maximize` for everything."""
    root = _harness_workspace(
        tmp_path / "ws", score=3.1, metric="wall_clock_s", direction="minimize"
    )

    assert HarnessValidator().validate(None, root, object()).direction == "minimize"


def test_the_harness_reports_one_number_and_no_second_opinion(tmp_path) -> None:
    """No leaderboard and no held-out set, so `_decide` must reach a verdict on
    the primary alone — the leaderboard branch is never taken."""
    root = _harness_workspace(tmp_path / "ws", score=0.5, metric="pass_rate", direction="maximize")

    assert HarnessValidator().validate(None, root, object()).secondary is None


def test_the_kaggle_search_would_misname_a_harness_metric(tmp_path) -> None:
    """Why the metric name has to travel *with* the score.

    `_primary_cv_keyed` ends its search on the generic `score` sentinel, so a
    harness blob does not come back empty — it comes back named `'score'`, which
    is not the name of anything. Two harness runs on different objectives would
    then both report `'score'` and `_same_metric` would match them, subtracting a
    pass rate from a wall-clock time.

    The result carrying `pass_rate` is what stops that, and it is the reason
    `metric` is a field rather than something re-derived from the blob.
    """
    from labpilot.research_engine.evidence.builder import _primary_cv_keyed

    root = _harness_workspace(tmp_path / "ws", score=0.82, metric="pass_rate", direction="maximize")
    result = HarnessValidator().validate(None, root, object())

    assert _primary_cv_keyed(result.raw) == (0.82, "score"), "the blob path names it wrongly"
    assert result.metric == "pass_rate", "the result names it correctly"


def test_the_protocol_is_satisfied_without_any_kaggle_machinery() -> None:
    """No competition, no knowledge dir, no metric registry. Anything this
    validator cannot do is something the seam genuinely requires."""
    validator: HypothesisValidator = HarnessValidator()

    assert validator.source == "harness"


# --- choosing between the two -----------------------------------------------


def test_a_result_file_selects_the_harness(tmp_path) -> None:
    root = _harness_workspace(tmp_path / "ws", score=1.0, metric="m", direction="maximize")

    assert handles(root)
    assert _validator_for(root).source == "harness"


def test_a_metrics_file_alongside_it_keeps_the_established_path(tmp_path) -> None:
    """A workspace holding both is ambiguous, and guessing which objective a
    campaign is judged on is the failure this milestone exists to remove. The
    established path wins rather than the newer one."""
    root = _harness_workspace(tmp_path / "ws", score=1.0, metric="m", direction="maximize")
    (root / "metrics.json").write_text(json.dumps({"cv_rmse": 1.0}), encoding="utf-8")

    assert not handles(root)
    assert _validator_for(root).source == "kaggle_cv"


def test_a_workspace_with_neither_is_still_kaggle(tmp_path) -> None:
    """Kaggle is the default because it is what every existing workspace is."""
    assert _validator_for(tmp_path).source == "kaggle_cv"


# --- a harness is somebody else's script, and will get this wrong -----------


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"metric": "m", "direction": "maximize"}, "no score reported"),
        ({"score": "0.8", "metric": "m", "direction": "maximize"}, "is not a number"),
        ({"score": True, "metric": "m", "direction": "maximize"}, "is not a number"),
    ],
)
def test_a_score_that_is_not_a_number_is_not_a_score(payload, expected) -> None:
    """`True` is an `int` in Python, so a `{"score": true}` would otherwise be
    scored as 1.0 — a harness bug worth naming rather than silently believing."""
    result = result_from_payload(payload)

    assert result.score is None
    assert any(expected in line for line in result.provenance)


def test_a_harness_that_does_not_name_its_metric_gets_no_name(tmp_path) -> None:
    """Which `_same_metric` then refuses to compare, rather than matching it
    against another unnamed score."""
    result = result_from_payload({"score": 0.9, "direction": "maximize"})

    assert result.metric == ""
    assert any("did not name the metric" in line for line in result.provenance)


def test_an_unstatable_direction_is_carried_as_unknown() -> None:
    """Not guessed. `build_evidence_card` is the layer that decides refusing is
    the right response to a missing sign."""
    result = result_from_payload({"score": 0.9, "metric": "m", "direction": "sideways"})

    assert result.direction is None
    assert any("sideways" in line for line in result.provenance)


@pytest.mark.parametrize("body", ["{ not json", "[1, 2, 3]", '"a string"', ""])
def test_a_malformed_result_file_refuses_rather_than_raising(tmp_path, body) -> None:
    """A traceback inside a comparison helps nobody; a result that says what was
    wrong reaches the operator."""
    root = tmp_path / "ws"
    root.mkdir()
    (root / RESULT_FILE).write_text(body, encoding="utf-8")

    result = HarnessValidator().validate(None, root, object())

    assert result.score is None and result.direction is None


def test_a_missing_result_file_is_not_a_crash(tmp_path) -> None:
    result = HarnessValidator().validate(None, tmp_path, object())

    assert result.score is None


# --- exit criterion: a campaign runs against it end to end ------------------


def _harness_context(tmp_path: Path, *, treatment: dict, control: dict) -> TaskContext:
    competition = "bench-demo"
    paths = ResearchPaths(tmp_path / "knowledge", competition).ensure()
    root = _harness_workspace(tmp_path / "ws", **treatment)
    now = datetime.now(UTC)
    plan = ResearchPlan(
        id="P-001",
        competition=competition,
        hypothesis_id="H-001",
        goal="make the benchmark pass more often",
        status=PlanStatus.READY,
        tasks=[ResearchTask(id="P-001-T01", plan_id="P-001", type=TaskType.COMPARE)],
        created_at=now,
        updated_at=now,
        metadata={
            "plan_kind": "delta",
            "parent_execution_id": "E-000",
            "parent_metrics": control,
        },
    )
    return TaskContext(
        plan=plan,
        task=plan.tasks[0],
        execution=ResearchExecution(id="E-001", plan_id="P-001", competition=competition),
        paths=paths,
        workspace_root=root,
        competition=competition,
    )


def test_a_campaign_compares_two_harness_runs_end_to_end(tmp_path) -> None:
    """Exit criterion 2, hermetic. No Kaggle credentials, no network, no
    `competition.json` — and a signed verdict comes out the other end."""
    card = run_compare_and_build_card(
        _harness_context(
            tmp_path,
            treatment={"score": 0.82, "metric": "pass_rate", "direction": "maximize"},
            control={"score": 0.70, "metric": "pass_rate", "direction": "maximize"},
        )
    )

    assert card.maximize is True
    assert card.observed.parent_cv == 0.70
    assert card.observed.treatment_cv == 0.82
    assert card.observed.cv_gain == pytest.approx(0.12)
    assert card.decision.value == "accepted"


def test_a_minimised_benchmark_reads_the_other_way_end_to_end(tmp_path) -> None:
    """The same campaign on a wall-clock objective. A drop is an improvement,
    and nothing on disk says so except the harness."""
    card = run_compare_and_build_card(
        _harness_context(
            tmp_path,
            treatment={"score": 3.1, "metric": "wall_clock_s", "direction": "minimize"},
            control={"score": 9.4, "metric": "wall_clock_s", "direction": "minimize"},
        )
    )

    assert card.maximize is False
    assert card.observed.cv_gain == pytest.approx(3.1 - 9.4)
    assert card.decision.value == "accepted", "a benchmark that got faster is an improvement"


def test_stability_is_unknown_rather_than_fabricated(tmp_path) -> None:
    """A harness reports one number and no fold spread. `UNKNOWN` is the honest
    answer, and `_decide` has to reach a verdict without it."""
    card = run_compare_and_build_card(
        _harness_context(
            tmp_path,
            treatment={"score": 0.82, "metric": "pass_rate", "direction": "maximize"},
            control={"score": 0.70, "metric": "pass_rate", "direction": "maximize"},
        )
    )

    assert card.observed.stability.value == "unknown"
    assert card.observed.parent_cv_std is None and card.observed.treatment_cv_std is None


def test_a_campaign_refuses_when_the_harness_states_no_direction(tmp_path) -> None:
    """The refusal survives into the new domain. A card whose sign is a guess is
    worse than no card, whatever computed the score."""
    with pytest.raises(ValueError, match="maximises or minimises"):
        run_compare_and_build_card(
            _harness_context(
                tmp_path,
                treatment={"score": 0.82, "metric": "pass_rate"},
                control={"score": 0.70, "metric": "pass_rate"},
            )
        )


def test_two_harness_runs_on_different_objectives_are_not_compared(tmp_path) -> None:
    """`_same_metric` is domain-neutral, so the guard that stopped an accuracy
    being subtracted from an RMSE also stops a pass rate being subtracted from a
    wall-clock time."""
    card = run_compare_and_build_card(
        _harness_context(
            tmp_path,
            treatment={"score": 0.82, "metric": "pass_rate", "direction": "maximize"},
            control={"score": 9.4, "metric": "wall_clock_s", "direction": "maximize"},
        )
    )

    assert card.observed.cv_gain is None
    assert card.decision_reason.startswith("metric_key_mismatch")


# --- exit criterion: the layers above the validator did not change ----------


def test_no_layer_above_the_validator_learned_about_harnesses() -> None:
    """Exit criterion 1, asserted rather than claimed.

    The plan's promise is that Conductor, policy, hypotheses and reflection are
    already domain-neutral and *should not need to change*. A grep is a weak
    test in general; here it is the right one, because the claim is precisely
    that these modules contain no mention of the new domain.
    """
    root = Path(__file__).resolve().parents[2] / "src" / "labpilot" / "research_engine"
    neutral = ["conductor", "reflection", "memory", "context", "planner"]

    # Imports, not the word "harness": two of these files already discuss
    # benchmark harnesses in prose, and prose is not coupling. What would be
    # coupling is any of them reaching into `validation` — the seam is supposed
    # to be invisible from up here.
    offenders = [
        str(path.relative_to(root))
        for name in neutral
        for path in (root / name).rglob("*.py")
        if "research_engine.validation" in path.read_text(encoding="utf-8")
    ]

    assert not offenders, f"the seam leaked upward into: {offenders}"


def test_the_evidence_card_is_built_by_the_same_funnel(tmp_path) -> None:
    """Both validators end in `build_evidence_card`. A harness result reaching a
    different code path would mean two verdict rules, which is what `_decide`
    exists to prevent."""
    result = HarnessValidator().validate(
        None,
        _harness_workspace(
            tmp_path / "ws", score=0.9, metric="pass_rate", direction="maximize"
        ),
        object(),
    )
    card = build_evidence_card(
        knowledge_dir=tmp_path,
        competition="bench",
        treatment_execution_id="E-2",
        control_execution_id="E-1",
        treatment_metrics={},
        result=result,
        control_result=result_from_payload(
            {"score": 0.7, "metric": "pass_rate", "direction": "maximize"}
        ),
        persist=False,
    )

    assert card.decision.value == "accepted"
    assert card.observed.cv_gain == pytest.approx(0.2)


# --- exit criterion 3: the same command works in both domains ---------------


def _preflight(root: Path):
    """The launch gate, as `research conduct` calls it."""
    import typer

    from labpilot.cli.conduct import _preflight_objective

    class _Workspace:
        def __init__(self, path: Path) -> None:
            self.root = path

    try:
        return _preflight_objective(_Workspace(root), "bench", assume_yes=True)
    except typer.Exit as exit_:
        assert exit_.exit_code == 2
        return None


def test_a_benchmark_campaign_can_start_at_all(tmp_path) -> None:
    """Exit criterion 3, and it did not hold before this phase.

    `research conduct` refuses to launch when it cannot justify the objective,
    and it could only read a `competition.json` — so every harness workspace was
    refused, with advice to *"set evaluation_metric in competition.json"*: a file
    it will never have. The same phrasing has to work in both domains, and it
    could not start a campaign in one of them.
    """
    root = _harness_workspace(
        tmp_path / "ws", score=0.82, metric="pass_rate", direction="maximize"
    )

    meta = _preflight(root)

    assert meta is not None, "the benchmark campaign was refused at launch"
    assert meta["objective_metric"] == "pass_rate"
    assert meta["objective_direction"] == "maximize"


def test_the_objective_is_read_before_any_run_has_happened(tmp_path) -> None:
    """The launch gate runs before anything is executed, so `result.json` does
    not exist yet. Selecting on the result alone read a fresh harness workspace
    as a Kaggle one, which is why the promise and the product are two files."""
    root = tmp_path / "ws"
    root.mkdir()
    (root / OBJECTIVE_FILE).write_text(
        json.dumps({"metric": "pass_rate", "direction": "maximize"}), encoding="utf-8"
    )

    assert not (root / RESULT_FILE).exists()
    assert handles(root), "a workspace that has not run yet is still a harness workspace"
    assert _preflight(root) is not None


def test_a_harness_metric_this_repo_cannot_compute_is_not_a_blocker(tmp_path) -> None:
    """`pass_rate` is not in the metric registry and `compute_metric` cannot
    produce it — both true, and both irrelevant. The harness reports its own
    number, so there is no proxy to fall back to and no silent substitution to
    prevent. That check exists for cross-validation choosing a stand-in metric,
    which never happens here."""
    from labpilot.research_engine.intelligence.competition.metric_vocabulary import is_scorable

    assert not is_scorable("pass_rate")
    root = _harness_workspace(
        tmp_path / "ws", score=0.82, metric="pass_rate", direction="maximize"
    )

    assert _preflight(root) is not None


def test_a_harness_that_declares_no_objective_is_still_refused(tmp_path) -> None:
    """The gate must not become a rubber stamp in the new domain. An unjustified
    objective blocks a benchmark campaign exactly as it blocks a competition."""
    root = tmp_path / "ws"
    root.mkdir()
    (root / OBJECTIVE_FILE).write_text(json.dumps({}), encoding="utf-8")

    assert _preflight(root) is None


def test_a_harness_declaring_no_direction_is_refused(tmp_path) -> None:
    """A score with no direction is a number whose sign is a coin flip, whatever
    domain produced it."""
    root = tmp_path / "ws"
    root.mkdir()
    (root / OBJECTIVE_FILE).write_text(json.dumps({"metric": "pass_rate"}), encoding="utf-8")

    assert _preflight(root) is None


def test_the_refusal_names_the_file_this_workspace_actually_has(tmp_path, capsys) -> None:
    """A refusal nobody can act on is a wall, and this gate exists to ask rather
    than to wall. Telling a benchmark operator to edit `competition.json` is the
    same domain leak as refusing them outright."""
    root = tmp_path / "ws"
    root.mkdir()
    (root / OBJECTIVE_FILE).write_text(json.dumps({}), encoding="utf-8")

    _preflight(root)
    out = capsys.readouterr().out

    assert OBJECTIVE_FILE in out
    assert "competition.json" not in out


def test_a_competition_workspace_still_gets_competition_advice(tmp_path, capsys) -> None:
    """The other half. Making the message domain-aware must not cost the domain
    it already served."""
    root = tmp_path / "ws"
    root.mkdir()
    (root / "competition.json").write_text(json.dumps({"slug": "d"}), encoding="utf-8")

    _preflight(root)
    out = capsys.readouterr().out

    assert "competition.json" in out
    assert OBJECTIVE_FILE not in out
