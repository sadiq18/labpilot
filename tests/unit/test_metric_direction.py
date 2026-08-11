"""Which way is better, and what happens when nobody said.

Every one of these is a regression test for a single production defect measured
on rogii 2026-08-07: `build_evidence_card` defaulted to ``maximize=True`` and no
caller passed anything, so on an MSE competition the engine recorded its only
genuine improvement as a rejection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from labpilot.research_engine.evidence.builder import build_evidence_card
from labpilot.research_engine.evidence.models import ClaimEvidenceKind, EvidenceDecision
from labpilot.research_engine.intelligence.competition.direction import resolve_maximize
from labpilot.research_engine.intelligence.paths import ResearchPaths


def _competition_json(root: Path, direction: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "competition.json").write_text(
        json.dumps({"metric": {"name": "mse", "direction": direction}}),
        encoding="utf-8",
    )


# --- resolution ------------------------------------------------------------


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        ("minimize", False),
        ("maximize", True),
        ("MINIMIZE", False),
        ("min", False),
        ("max", True),
    ],
)
def test_direction_is_read_from_the_workspace(tmp_path, direction, expected):
    _competition_json(tmp_path, direction)
    assert resolve_maximize(competition="c", workspace_root=tmp_path) is expected


def test_an_absent_direction_is_unknown_not_maximize(tmp_path):
    """The whole defect in one assertion: silence must not read as 'maximize'."""
    (tmp_path / "competition.json").write_text(json.dumps({"metric": {}}), encoding="utf-8")
    assert resolve_maximize(competition="c", workspace_root=tmp_path) is None


def test_nothing_on_disk_is_unknown(tmp_path):
    assert resolve_maximize(competition="c", workspace_root=tmp_path) is None


def test_unparseable_direction_is_unknown(tmp_path):
    _competition_json(tmp_path, "whichever way is nicer")
    assert resolve_maximize(competition="c", workspace_root=tmp_path) is None


def test_the_workspace_copy_wins_over_the_knowledge_copy(tmp_path):
    """Nearest-first: the run is oriented by the file the run was given."""
    ws, kb = tmp_path / "ws", tmp_path / "kb"
    _competition_json(ws, "minimize")
    _competition_json(kb, "maximize")
    assert resolve_maximize(competition="c", workspace_root=ws, knowledge_root=kb) is False


def test_the_knowledge_copy_is_used_when_the_workspace_has_none(tmp_path):
    ws, kb = tmp_path / "ws", tmp_path / "kb"
    ws.mkdir()
    _competition_json(kb, "minimize")
    assert resolve_maximize(competition="c", workspace_root=ws, knowledge_root=kb) is False


def test_the_analyze_profile_artifact_is_the_last_resort(tmp_path):
    """This is where rogii's `minimize` actually lived, unread, the whole time."""
    extracted = tmp_path / "extracted" / "misc"
    extracted.mkdir(parents=True)
    (extracted / "competition_rogii.json").write_text(
        json.dumps({"metadata": {"profile": {"metric": {"name": "mse", "direction": "minimize"}}}}),
        encoding="utf-8",
    )
    got = resolve_maximize(competition="rogii", extracted_dir=tmp_path / "extracted")
    assert got is False


def test_corrupt_json_does_not_crash_the_caller(tmp_path):
    (tmp_path / "competition.json").write_text("{not json", encoding="utf-8")
    assert resolve_maximize(competition="c", workspace_root=tmp_path) is None


# --- the builder refuses to guess ------------------------------------------


def test_building_a_card_without_a_direction_raises(tmp_path):
    """Refusing to write beats writing a conclusion with an unknown sign."""
    with pytest.raises(ValueError, match="maximises or minimises"):
        build_evidence_card(
            knowledge_dir=tmp_path,
            competition="no-profile",
            treatment_execution_id="E-1",
            treatment_metrics={"cv_accuracy": 0.9},
            persist=False,
        )


def test_an_explicit_direction_still_wins(tmp_path):
    card = build_evidence_card(
        knowledge_dir=tmp_path,
        competition="no-profile",
        treatment_execution_id="E-1",
        treatment_metrics={"cv_accuracy": 0.9},
        maximize=False,
        persist=False,
    )
    assert card.maximize is False


def test_direction_resolves_from_the_knowledge_tree(tmp_path):
    competition = "resolves"
    paths = ResearchPaths(tmp_path, competition).ensure()
    _competition_json(paths.root, "minimize")
    card = build_evidence_card(
        knowledge_dir=tmp_path,
        competition=competition,
        treatment_execution_id="E-1",
        treatment_metrics={"cv_mse": 1.0},
        persist=False,
    )
    assert card.maximize is False


# --- the rogii inversion, end to end ---------------------------------------


def test_an_mse_improvement_is_accepted_not_rejected(tmp_path):
    """EV-012: SWA cut MSE 194.80 -> 190.97 and was recorded `rejected`."""
    competition = "rogii-like"
    paths = ResearchPaths(tmp_path, competition).ensure()
    _competition_json(paths.root, "minimize")
    card = build_evidence_card(
        knowledge_dir=tmp_path,
        competition=competition,
        treatment_execution_id="E-treat",
        treatment_metrics={"cv_mse": 190.97471945924474, "cv_std": 0.01},
        control_execution_id="E-ctrl",
        control_metrics={"cv_mse": 194.80084243002463, "cv_std": 0.01},
        persist=False,
    )
    assert card.maximize is False
    assert card.observed.cv_gain is not None and card.observed.cv_gain < 0
    assert card.decision == EvidenceDecision.ACCEPTED, card.decision_reason


def test_an_mse_regression_is_rejected_not_accepted(tmp_path):
    """EV-015: the metric got worse and the card said `accepted`."""
    competition = "rogii-like"
    paths = ResearchPaths(tmp_path, competition).ensure()
    _competition_json(paths.root, "minimize")
    card = build_evidence_card(
        knowledge_dir=tmp_path,
        competition=competition,
        treatment_execution_id="E-treat",
        treatment_metrics={"cv_mse": 194.80, "cv_std": 0.01},
        control_execution_id="E-ctrl",
        control_metrics={"cv_mse": 194.34, "cv_std": 0.01},
        persist=False,
    )
    assert card.decision == EvidenceDecision.REJECTED, card.decision_reason


def test_the_claim_verb_follows_the_direction(tmp_path):
    """A minimised metric improving must not be written up as 'hurts'.

    A real hypothesis is created because attribution is keyed off it: without
    one there are no techniques, no claim updates, and the assertion below would
    pass on an empty list while proving nothing.
    """
    from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore

    competition = "rogii-like"
    paths = ResearchPaths(tmp_path, competition).ensure()
    _competition_json(paths.root, "minimize")
    hyp = HypothesisStore(tmp_path, competition).create(
        observation="o", reason="r", prediction="p", confidence=0.7, technique="SWA"
    )
    card = build_evidence_card(
        knowledge_dir=tmp_path,
        competition=competition,
        treatment_execution_id="E-treat",
        treatment_metrics={"cv_mse": 190.97, "cv_std": 0.01},
        control_execution_id="E-ctrl",
        control_metrics={"cv_mse": 194.80, "cv_std": 0.01},
        hypothesis_id=hyp.id,
        persist=False,
    )
    verbs = [u.claim for u in card.claim_updates]
    assert verbs, "no claim updates: the test would prove nothing"
    assert all("improves" in v for v in verbs), verbs
    # The sentence is the half a human reads; `evidence` and `confidence_delta`
    # are the half that steers the belief store. Asserting only the verb let an
    # inverted polarity through review once already.
    assert all(u.evidence == ClaimEvidenceKind.SUPPORT for u in card.claim_updates)
    assert all(u.confidence_delta > 0 for u in card.claim_updates)


def test_belief_polarity_follows_the_direction_too(tmp_path):
    """An MSE improvement must not teach the belief store that SWA is harmful.

    `apply_card_to_beliefs` keys both the confidence step and the recorded
    `effect` off `evidence`, so a card can say "improves" while pushing the
    belief the other way. That is what happens when only the verb is oriented.
    """
    from labpilot.research_engine.evidence.apply import apply_card_to_beliefs
    from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
    from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore

    competition = "rogii-like"
    paths = ResearchPaths(tmp_path, competition).ensure()
    _competition_json(paths.root, "minimize")
    hyp = HypothesisStore(tmp_path, competition).create(
        observation="o", reason="r", prediction="p", confidence=0.7, technique="SWA"
    )
    card = build_evidence_card(
        knowledge_dir=tmp_path,
        competition=competition,
        treatment_execution_id="E-treat",
        treatment_metrics={"cv_mse": 190.97, "cv_std": 0.01},
        control_execution_id="E-ctrl",
        control_metrics={"cv_mse": 194.80, "cv_std": 0.01},
        hypothesis_id=hyp.id,
        persist=True,
    )
    apply_card_to_beliefs(knowledge_dir=tmp_path, competition=competition, card=card)

    with KnowledgeStore(tmp_path, competition) as store:
        beliefs = {b["technique"]: b for b in store.list_beliefs()}
    swa = beliefs.get("SWA")
    assert swa is not None, beliefs
    assert swa["effect"] == "positive", swa
    assert float(swa["confidence"]) > 0.5, swa


def test_a_regression_still_lowers_the_belief(tmp_path):
    """The mirror case, so the fix cannot be 'always positive'."""
    from labpilot.research_engine.evidence.apply import apply_card_to_beliefs
    from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
    from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore

    competition = "rogii-like"
    paths = ResearchPaths(tmp_path, competition).ensure()
    _competition_json(paths.root, "minimize")
    hyp = HypothesisStore(tmp_path, competition).create(
        observation="o", reason="r", prediction="p", confidence=0.7, technique="vit"
    )
    card = build_evidence_card(
        knowledge_dir=tmp_path,
        competition=competition,
        treatment_execution_id="E-treat",
        treatment_metrics={"cv_mse": 250.0, "cv_std": 0.01},
        control_execution_id="E-ctrl",
        control_metrics={"cv_mse": 194.80, "cv_std": 0.01},
        hypothesis_id=hyp.id,
        persist=True,
    )
    assert card.decision == EvidenceDecision.REJECTED
    assert all("hurts" in u.claim for u in card.claim_updates)
    apply_card_to_beliefs(knowledge_dir=tmp_path, competition=competition, card=card)

    with KnowledgeStore(tmp_path, competition) as store:
        beliefs = {b["technique"]: b for b in store.list_beliefs()}
    assert beliefs["vit"]["effect"] == "negative", beliefs["vit"]
    assert float(beliefs["vit"]["confidence"]) < 0.5, beliefs["vit"]


# --- the campaign's own compass ---------------------------------------------


def test_campaign_budget_resolves_direction_from_the_competition(tmp_path):
    """`BudgetConfig.maximize` defaulted True with nothing overriding it.

    rogii session S-018 stored `"maximize": true` on an MSE competition. Latent
    only because that run had no metric target — with one, the campaign would
    have stopped on the wrong side.
    """
    from labpilot.cli.conduct import _resolve_campaign_direction

    competition = "budget-demo"
    paths = ResearchPaths(tmp_path, competition).ensure()
    _competition_json(paths.root, "minimize")

    class _WS:
        knowledge_dir = tmp_path
        root = tmp_path / "ws"

    assert _resolve_campaign_direction(_WS(), competition) is False


def test_an_unknown_direction_leaves_the_budget_default_alone(tmp_path):
    from labpilot.cli.conduct import _resolve_campaign_direction

    ResearchPaths(tmp_path, "no-profile").ensure()

    class _WS:
        knowledge_dir = tmp_path
        root = tmp_path / "ws"

    assert _resolve_campaign_direction(_WS(), "no-profile") is None


def test_objective_check_agrees_with_the_budget_default():
    """loop.py read `maximize` defaulting False while BudgetConfig defaults True."""
    from labpilot.research_engine.conductor.budgets import BudgetConfig
    from labpilot.research_engine.conductor.loop import _objective_unmet

    class _BareConfig:
        target_metric = "mse"
        target_value = 100.0

    class _State:
        last_metric = 50.0

    default = BudgetConfig(target_metric="mse", target_value=100.0)
    assert _objective_unmet(default, _State()) == _objective_unmet(_BareConfig(), _State())


def test_a_minimised_target_is_met_by_going_below_it():
    from labpilot.research_engine.conductor.budgets import BudgetConfig
    from labpilot.research_engine.conductor.loop import _objective_unmet

    cfg = BudgetConfig(target_metric="mse", target_value=100.0, maximize=False)

    class _Below:
        last_metric = 50.0

    class _Above:
        last_metric = 150.0

    assert _objective_unmet(cfg, _Below()) is False
    assert _objective_unmet(cfg, _Above()) is True


def test_a_target_with_an_unknown_direction_is_refused(tmp_path):
    """A target is the only thing that reads `maximize`, so an unknown
    direction is harmless until one is set — and unacceptable after.

    Leaving BudgetConfig's `True` default in place would stop the campaign on
    the wrong side. `build_evidence_card` already refuses on this; the campaign
    should not be more permissive about the same unknown.
    """
    import typer

    from labpilot.cli.conduct import _budget_metadata

    with pytest.raises(typer.BadParameter, match="maximised or minimised"):
        _budget_metadata(
            max_submissions=None,
            max_wall_s=None,
            max_cost_usd=None,
            target_metric="mse",
            target_value=5.0,
            plateau_window=3,
            maximize=None,
        )


def _spec_competition_json(root: Path, direction: str | None) -> None:
    """The shape `CompetitionParser.save` writes — a `CompetitionSpec`, whose
    metric is `evaluation_metric` rather than `metric`."""
    root.mkdir(parents=True, exist_ok=True)
    metric: dict[str, object] = {"name": "mse", "key": "mse"}
    if direction is not None:
        metric["direction"] = direction
    (root / "competition.json").write_text(
        json.dumps({"slug": "demo", "title": "demo", "evaluation_metric": metric}),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("direction", "expected"), [("minimize", False), ("maximize", True), ("Minimize", False)]
)
def test_a_parser_written_spec_resolves(tmp_path, direction, expected):
    """`CompetitionParser.save` writes `evaluation_metric`, not `metric`.

    Reading only `metric` meant every machine-generated spec — most of them —
    answered "unknown" and fell through to the profile artifact, so a
    competition with an explicit direction on disk was still unresolvable.
    """
    _spec_competition_json(tmp_path, direction)

    assert resolve_maximize(competition="demo", workspace_root=tmp_path) is expected


def test_a_parser_written_spec_without_a_direction_stays_unknown(tmp_path):
    """`MetricSpec.direction` defaults to "maximize" in the model, so this is
    read as a dict — parsing through `CompetitionSpec` would turn an absent
    field into a confident wrong answer instead of the None that lets the
    caller keep looking."""
    _spec_competition_json(tmp_path, None)

    assert resolve_maximize(competition="demo", workspace_root=tmp_path) is None


def test_the_hand_written_metric_block_keeps_its_veto(tmp_path):
    """When both shapes are present the hand-written `metric` decides,
    including when its answer is "unknown".

    Falling through on an unparseable direction would let a generated
    `evaluation_metric` override a deliberate `metric` whose direction is
    merely misspelled.
    """
    (tmp_path / "competition.json").write_text(
        json.dumps(
            {
                "metric": {"name": "mse", "direction": "lower-is-better"},
                "evaluation_metric": {"name": "mse", "key": "mse", "direction": "maximize"},
            }
        ),
        encoding="utf-8",
    )

    assert resolve_maximize(competition="demo", workspace_root=tmp_path) is None


def test_a_parser_written_spec_resolves_from_the_knowledge_copy(tmp_path):
    """The caller-visible consequence: `_resolve_direction` refuses to sign a
    conclusion it cannot orient, so widening the reader turns a refusal into a
    built card. Pin it, since it is a behaviour change for four callers."""
    paths = ResearchPaths(tmp_path / "kb", "demo")
    _spec_competition_json(paths.root, "minimize")

    assert resolve_maximize(competition="demo", knowledge_root=paths.root) is False


def test_no_target_tolerates_an_unknown_direction(tmp_path):
    """Nothing reads `maximize` without a target, so this must not block a run."""
    from labpilot.cli.conduct import _budget_metadata

    meta = _budget_metadata(
        max_submissions=None,
        max_wall_s=None,
        max_cost_usd=None,
        target_metric=None,
        target_value=None,
        plateau_window=3,
        maximize=None,
    )
    assert meta["budgets"]["target_value"] is None
