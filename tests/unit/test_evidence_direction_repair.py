"""Cards written with an inverted compass must heal, not persist.

Repair exists because 15 rogii cards were already on disk when the direction bug
was found, and the standing rule is that labpilot fixes its own memory rather
than the user hand-editing workspace artifacts.
"""

from __future__ import annotations

import json

import pytest

from labpilot.research_engine.evidence.models import (
    EvidenceCard,
    EvidenceDecision,
    ObservedOutcomes,
)
from labpilot.research_engine.evidence.repair import repair_card_directions
from labpilot.research_engine.evidence.store import EvidenceCardStore
from labpilot.research_engine.intelligence.paths import ResearchPaths

COMPETITION = "repair-demo"


@pytest.fixture
def knowledge(tmp_path):
    paths = ResearchPaths(tmp_path, COMPETITION).ensure()
    (paths.root / "competition.json").write_text(
        json.dumps({"metric": {"name": "mse", "direction": "minimize"}}), encoding="utf-8"
    )
    return tmp_path


def _save(knowledge_dir, *, parent, treatment, decision, maximize=True, attribution=None):
    store = EvidenceCardStore(knowledge_dir, COMPETITION)
    return store.save(
        EvidenceCard(
            competition=COMPETITION,
            control_experiment="E-ctrl",
            treatment_experiment="E-treat",
            observed=ObservedOutcomes(
                cv_gain=treatment - parent, parent_cv=parent, treatment_cv=treatment
            ),
            technique_attribution=attribution or {},
            decision=decision,
            decision_reason="original",
            maximize=maximize,
        )
    )


def test_an_inverted_rejection_becomes_an_acceptance(knowledge):
    """EV-012 exactly: SWA cut MSE 194.80 -> 190.97, recorded `rejected`."""
    card = _save(
        knowledge,
        parent=194.80084243002463,
        treatment=190.97471945924474,
        decision=EvidenceDecision.REJECTED,
        attribution={"SWA": -3.826122970779892},
    )
    assert repair_card_directions(knowledge, COMPETITION) == [card.id]
    fixed = EvidenceCardStore(knowledge, COMPETITION).get(card.id)
    assert fixed.maximize is False
    assert fixed.decision == EvidenceDecision.ACCEPTED
    assert "re-oriented" in fixed.decision_reason


def test_an_inverted_acceptance_becomes_a_rejection(knowledge):
    """EV-015: the metric got worse and the card said `accepted`."""
    card = _save(
        knowledge, parent=194.34, treatment=194.80, decision=EvidenceDecision.ACCEPTED
    )
    repair_card_directions(knowledge, COMPETITION)
    fixed = EvidenceCardStore(knowledge, COMPETITION).get(card.id)
    assert fixed.decision == EvidenceDecision.REJECTED


def test_the_claim_verb_is_re_oriented_too(knowledge):
    """A repaired verdict with a stale verb would still read backwards."""
    card = _save(
        knowledge,
        parent=194.80,
        treatment=190.97,
        decision=EvidenceDecision.REJECTED,
        attribution={"SWA": -3.83},
    )
    repair_card_directions(knowledge, COMPETITION)
    fixed = EvidenceCardStore(knowledge, COMPETITION).get(card.id)
    verbs = [u.claim for u in fixed.claim_updates]
    assert verbs and all("improves" in v for v in verbs), verbs


def test_measurements_are_never_rewritten(knowledge):
    """Repair changes what the numbers *mean*, never the numbers."""
    card = _save(
        knowledge, parent=194.80, treatment=190.97, decision=EvidenceDecision.REJECTED
    )
    repair_card_directions(knowledge, COMPETITION)
    fixed = EvidenceCardStore(knowledge, COMPETITION).get(card.id)
    assert fixed.observed.parent_cv == card.observed.parent_cv
    assert fixed.observed.treatment_cv == card.observed.treatment_cv
    assert fixed.observed.cv_gain == card.observed.cv_gain


def test_correctly_oriented_cards_are_left_alone(knowledge):
    card = _save(
        knowledge,
        parent=194.80,
        treatment=190.97,
        decision=EvidenceDecision.ACCEPTED,
        maximize=False,
    )
    assert repair_card_directions(knowledge, COMPETITION) == []
    unchanged = EvidenceCardStore(knowledge, COMPETITION).get(card.id)
    assert unchanged.decision_reason == "original"


def test_repair_is_idempotent(knowledge):
    _save(knowledge, parent=194.80, treatment=190.97, decision=EvidenceDecision.REJECTED)
    first = repair_card_directions(knowledge, COMPETITION)
    assert first
    assert repair_card_directions(knowledge, COMPETITION) == []


def test_an_unknown_direction_repairs_nothing(tmp_path):
    """Rewriting on a guess would be the original defect in new clothes."""
    ResearchPaths(tmp_path, COMPETITION).ensure()
    _save(tmp_path, parent=194.80, treatment=190.97, decision=EvidenceDecision.REJECTED)
    assert repair_card_directions(tmp_path, COMPETITION) == []


def test_a_missing_store_does_not_break_a_campaign(tmp_path):
    assert repair_card_directions(tmp_path / "nope", COMPETITION) == []
