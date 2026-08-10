"""apply_card_to_hypothesis must not bypass HypothesisStore's lock (M11).

It used to call `store.update_outcome(...)` (locked) then, separately, read
the hypothesis again and write the confidence bump via `store._save(...)`
directly — a second, unlocked read-modify-write racing everything else this
class locks against. Fixed by folding the confidence bump into the single
`update_outcome(...)` call.
"""

from __future__ import annotations

from pathlib import Path

from labpilot.research_engine.evidence.apply import apply_card_to_hypothesis
from labpilot.research_engine.evidence.models import (
    EvidenceCard,
    EvidenceDecision,
    ObservedOutcomes,
)
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.shared.experiments.models import HypothesisStatus


def test_apply_card_bumps_confidence_and_status_in_one_call(tmp_path: Path) -> None:
    store = HypothesisStore(tmp_path / "knowledge", "titanic")
    hyp = store.create(observation="a", reason="b", prediction="c", confidence=0.5)

    card = EvidenceCard(
        id="EV-001",
        competition="titanic",
        hypothesis_id=hyp.id,
        decision=EvidenceDecision.ACCEPTED,
        observed=ObservedOutcomes(cv_gain=0.01),
        treatment_experiment="run-1",
    )

    apply_card_to_hypothesis(
        knowledge_dir=tmp_path / "knowledge",
        competition="titanic",
        card=card,
    )

    updated = store.get(hyp.id)
    assert updated is not None
    assert updated.status == HypothesisStatus.CONFIRMED
    assert updated.confidence == 0.55  # +0.05 nudge for ACCEPTED
    assert updated.actual_outcome is not None


def test_apply_card_rejected_lowers_confidence(tmp_path: Path) -> None:
    store = HypothesisStore(tmp_path / "knowledge", "titanic")
    hyp = store.create(observation="a", reason="b", prediction="c", confidence=0.5)

    card = EvidenceCard(
        id="EV-001",
        competition="titanic",
        hypothesis_id=hyp.id,
        decision=EvidenceDecision.REJECTED,
        observed=ObservedOutcomes(),
        treatment_experiment="run-1",
    )

    apply_card_to_hypothesis(
        knowledge_dir=tmp_path / "knowledge",
        competition="titanic",
        card=card,
    )

    updated = store.get(hyp.id)
    assert updated is not None
    assert updated.status == HypothesisStatus.REJECTED
    assert updated.confidence == 0.45  # -0.05 nudge for REJECTED
