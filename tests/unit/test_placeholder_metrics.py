"""A run that never trained a model must not become evidence.

`training/capability.py` writes ``{"cv_accuracy": 0.5, "status": "dry_run_stub"}``
on a dry run, and the generated fallback script writes ``{"cv_accuracy": 0.0,
"status": "last_resort_scaffold"}``. Both markers were present the whole time and
nothing read them, so on rogii seven of fifteen evidence cards were built from
runs that trained nothing — including EV-001, the sole basis of the false claim
"vit improves the primary metric".
"""

from __future__ import annotations

import json

import pytest

from labpilot.research_engine.evidence.builder import (
    build_evidence_card,
    is_placeholder_metrics,
)
from labpilot.research_engine.evidence.models import (
    EvidenceCard,
    EvidenceDecision,
    ObservedOutcomes,
)
from labpilot.research_engine.evidence.repair import repair_card_directions
from labpilot.research_engine.evidence.store import EvidenceCardStore
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.intelligence.models import (
    ResearchArtifact,
    ResearchArtifactType,
)
from labpilot.research_engine.intelligence.paths import ResearchPaths

COMPETITION = "placeholder-demo"


@pytest.fixture
def knowledge(tmp_path):
    paths = ResearchPaths(tmp_path, COMPETITION).ensure()
    (paths.root / "competition.json").write_text(
        json.dumps({"metric": {"name": "mse", "direction": "minimize"}}), encoding="utf-8"
    )
    return tmp_path


# --- recognising the marker -------------------------------------------------


@pytest.mark.parametrize("status", ["dry_run_stub", "last_resort_scaffold"])
def test_the_real_markers_are_recognised(status):
    assert is_placeholder_metrics({"cv_accuracy": 0.5, "status": status})


def test_case_and_whitespace_do_not_defeat_it():
    assert is_placeholder_metrics({"status": "  Dry_Run_Stub  "})


@pytest.mark.parametrize("metrics", [None, {}, {"cv_mse": 1.0}, {"status": "ok"}])
def test_real_runs_are_not_flagged(metrics):
    assert not is_placeholder_metrics(metrics)


# --- refusing to mint -------------------------------------------------------


def test_a_stub_treatment_does_not_produce_a_verdict(knowledge):
    """EV-002..007: MSE 194.80 -> 0.50 read as an enormous improvement."""
    card = build_evidence_card(
        knowledge_dir=knowledge,
        competition=COMPETITION,
        treatment_execution_id="E-stub",
        treatment_metrics={"cv_accuracy": 0.5, "status": "dry_run_stub"},
        control_execution_id="E-real",
        control_metrics={"cv_mse": 194.80},
        persist=False,
    )
    assert card.decision == EvidenceDecision.INCONCLUSIVE
    assert "placeholder_metrics" in card.decision_reason
    assert card.observed.cv_gain is None


def test_a_scaffold_control_does_not_produce_a_verdict(knowledge):
    """EV-001: the vit card, whose control was a scaffold scoring 0.0."""
    card = build_evidence_card(
        knowledge_dir=knowledge,
        competition=COMPETITION,
        treatment_execution_id="E-real",
        treatment_metrics={"cv_mse": 194.80},
        control_execution_id="E-scaffold",
        control_metrics={"cv_accuracy": 0.0, "status": "last_resort_scaffold"},
        persist=False,
    )
    assert card.decision == EvidenceDecision.INCONCLUSIVE
    assert "placeholder_metrics" in card.decision_reason


def test_no_claim_is_minted_from_a_placeholder(knowledge):
    card = build_evidence_card(
        knowledge_dir=knowledge,
        competition=COMPETITION,
        treatment_execution_id="E-stub",
        treatment_metrics={"cv_accuracy": 0.5, "status": "dry_run_stub"},
        control_execution_id="E-real",
        control_metrics={"cv_mse": 194.80},
        persist=False,
    )
    assert card.claim_updates == []


def test_two_real_runs_still_get_a_verdict(knowledge):
    """The guard must not swallow genuine comparisons."""
    card = build_evidence_card(
        knowledge_dir=knowledge,
        competition=COMPETITION,
        treatment_execution_id="E-b",
        treatment_metrics={"cv_mse": 190.97, "cv_std": 0.01},
        control_execution_id="E-a",
        control_metrics={"cv_mse": 194.80, "cv_std": 0.01},
        persist=False,
    )
    assert card.decision == EvidenceDecision.ACCEPTED


# --- mismatched metrics -----------------------------------------------------


def test_an_accuracy_is_never_subtracted_from_an_rmse(knowledge):
    """Different metrics are not comparable even when both runs are real."""
    card = build_evidence_card(
        knowledge_dir=knowledge,
        competition=COMPETITION,
        treatment_execution_id="E-b",
        treatment_metrics={"cv_accuracy": 0.91},
        control_execution_id="E-a",
        control_metrics={"cv_rmse": 194.80},
        persist=False,
    )
    assert card.decision == EvidenceDecision.INCONCLUSIVE
    assert "metric_key_mismatch" in card.decision_reason
    assert card.observed.cv_gain is None


# --- retiring what was already written --------------------------------------


def _execution(store, exec_id, metrics):
    store.upsert_artifact(
        ResearchArtifact(
            id=f"exp:execution:{exec_id}",
            type=ResearchArtifactType.EXPERIMENT,
            source="labpilot",
            title=exec_id,
            competition_slug=COMPETITION,
            metadata={"execution_id": exec_id, "metrics": metrics},
        )
    )


def test_a_legacy_placeholder_card_is_retired(knowledge):
    """Identified from the execution artifact, not guessed from its scores."""
    with KnowledgeStore(knowledge, COMPETITION) as store:
        _execution(store, "E-stub", {"cv_accuracy": 0.5, "status": "dry_run_stub"})
        _execution(store, "E-real", {"cv_mse": 194.80})

    cards = EvidenceCardStore(knowledge, COMPETITION)
    card = cards.save(
        EvidenceCard(
            competition=COMPETITION,
            control_experiment="E-real",
            treatment_experiment="E-stub",
            observed=ObservedOutcomes(cv_gain=-194.3, parent_cv=194.80, treatment_cv=0.5),
            technique_attribution={"hyp:H-010": -194.3},
            decision=EvidenceDecision.REJECTED,
            decision_reason="cv_gain_negative",
            maximize=True,
        )
    )
    assert repair_card_directions(knowledge, COMPETITION) == [card.id]
    fixed = cards.get(card.id)
    assert fixed.decision == EvidenceDecision.INCONCLUSIVE
    assert "placeholder_metrics" in fixed.decision_reason
    assert fixed.claim_updates == []


def test_retiring_is_idempotent(knowledge):
    with KnowledgeStore(knowledge, COMPETITION) as store:
        _execution(store, "E-stub", {"cv_accuracy": 0.5, "status": "dry_run_stub"})

    EvidenceCardStore(knowledge, COMPETITION).save(
        EvidenceCard(
            competition=COMPETITION,
            control_experiment="E-real",
            treatment_experiment="E-stub",
            observed=ObservedOutcomes(cv_gain=-194.3, parent_cv=194.80, treatment_cv=0.5),
            decision=EvidenceDecision.REJECTED,
            maximize=True,
        )
    )
    assert repair_card_directions(knowledge, COMPETITION)
    assert repair_card_directions(knowledge, COMPETITION) == []


def test_a_real_card_is_re_oriented_not_retired(knowledge):
    """Retirement must not become a blunt instrument that erases real results."""
    with KnowledgeStore(knowledge, COMPETITION) as store:
        _execution(store, "E-a", {"cv_mse": 194.80})
        _execution(store, "E-b", {"cv_mse": 190.97})

    cards = EvidenceCardStore(knowledge, COMPETITION)
    card = cards.save(
        EvidenceCard(
            competition=COMPETITION,
            control_experiment="E-a",
            treatment_experiment="E-b",
            observed=ObservedOutcomes(cv_gain=-3.83, parent_cv=194.80, treatment_cv=190.97),
            technique_attribution={"SWA": -3.83},
            decision=EvidenceDecision.REJECTED,
            maximize=True,
        )
    )
    repair_card_directions(knowledge, COMPETITION)
    fixed = cards.get(card.id)
    assert fixed.decision == EvidenceDecision.ACCEPTED
    assert "re-oriented" in fixed.decision_reason
