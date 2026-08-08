"""An overlay must not keep teaching a verdict its card has since reversed.

The real record, from rogii 2026-08-08. E-026 is the only execution that ever
improved the metric (MSE 194.80 -> 190.97). `repair_card_directions` had already
re-oriented its card to `accepted`; every overlay still said:

    - Avoid: SWA
    - Avoid: regression on E-026

`upsert_skill_overlay` returns early once a `lesson_id` is present, so nothing
would ever have corrected it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from labpilot.research_engine.evidence.models import EvidenceCard, EvidenceDecision
from labpilot.research_engine.evidence.overlay_repair import (
    record_references_in_overlays,
    repair_skill_overlays,
)
from labpilot.research_engine.evidence.store import EvidenceCardStore

_KNOWLEDGE = "knowledge"
_COMP = "rogii-wellbore-geology-prediction"


def _overlay(root: Path, text: str, name: str = "code_engineer.md") -> Path:
    path = root / ".labpilot" / "skills" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _card(root: Path, execution: str, decision: EvidenceDecision) -> EvidenceCard:
    store = EvidenceCardStore(root / _KNOWLEDGE, _COMP)
    return store.save(
        EvidenceCard(
            competition=_COMP,
            treatment_experiment=execution,
            decision=decision,
        )
    )


def _repair(root: Path) -> list[str]:
    return repair_skill_overlays(root, root / _KNOWLEDGE, _COMP)


_SWA_LESSON = """<!-- lesson:E-026 -->
## Lesson `E-026`
- Avoid: SWA
- Avoid: regression on E-026
- Try: Investigate: anchor_target leakage
- Note: CV delta -3.8261; LB delta None
"""


def test_the_only_real_improvement_stops_being_taught_as_a_regression(tmp_path):
    path = _overlay(tmp_path, _SWA_LESSON)
    _card(tmp_path, "E-026", EvidenceDecision.ACCEPTED)

    assert _repair(tmp_path) == ["code_engineer.md"]
    text = path.read_text(encoding="utf-8")

    assert "- Keep: SWA" in text
    assert "Avoid: SWA" not in text


def test_the_regression_claim_is_dropped_rather_than_inverted(tmp_path):
    """ "regression on E-026" is only true on the Avoid side.

    Flipped verbatim it would read as a reason to *keep* the thing it says
    regressed — a sentence corrected at the label and left wrong in the body,
    which is the "compass had two needles" failure.
    """
    path = _overlay(tmp_path, _SWA_LESSON)
    _card(tmp_path, "E-026", EvidenceDecision.ACCEPTED)
    _repair(tmp_path)

    assert "regression on E-026" not in path.read_text(encoding="utf-8")


def test_a_card_that_now_rejects_flips_keep_to_avoid(tmp_path):
    """E-030 was recorded `accepted`/`strong` and was a genuine regression."""
    path = _overlay(
        tmp_path,
        "<!-- lesson:E-030 -->\n## Lesson `E-030`\n- Keep: rolling_features\n",
    )
    _card(tmp_path, "E-030", EvidenceDecision.REJECTED)
    _repair(tmp_path)

    assert "- Avoid: rolling_features" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("decision", [EvidenceDecision.INCONCLUSIVE, None])
def test_a_lesson_with_no_conclusive_card_is_dropped_not_restated(tmp_path, decision):
    """Includes the `dry_run_stub` runs that trained no model — how `Keep: vit`
    survived. There is nothing to restate it from, and inventing a verdict is
    the failure this module removes."""
    path = _overlay(tmp_path, "<!-- lesson:E-004 -->\n## Lesson `E-004`\n- Keep: vit\n")
    # A card must exist for the store to be non-empty, but not for E-004.
    _card(tmp_path, "E-999", EvidenceDecision.ACCEPTED)
    if decision is not None:
        _card(tmp_path, "E-004", decision)
    _repair(tmp_path)

    assert "vit" not in path.read_text(encoding="utf-8")


def test_repair_is_idempotent(tmp_path):
    """Result is a function of the cards alone, so a second run changes nothing."""
    _overlay(tmp_path, _SWA_LESSON)
    _card(tmp_path, "E-026", EvidenceDecision.ACCEPTED)

    assert _repair(tmp_path) == ["code_engineer.md"]
    assert _repair(tmp_path) == []


def test_every_agent_overlay_is_rebuilt(tmp_path):
    """One lesson is broadcast to six agents, so repair must reach all six."""
    names = [
        "code_engineer.md",
        "hypothesis_generator.md",
        "research_planner.md",
        "planning_engine.md",
        "experiment_reviewer.md",
        "research_brief.md",
    ]
    for name in names:
        _overlay(tmp_path, _SWA_LESSON, name)
    _card(tmp_path, "E-026", EvidenceDecision.ACCEPTED)

    assert sorted(_repair(tmp_path)) == sorted(names)


def test_no_cards_means_no_rewrite(tmp_path):
    """Without cards there is nothing to correct towards — leave it alone."""
    path = _overlay(tmp_path, _SWA_LESSON)
    before = path.read_text(encoding="utf-8")

    assert _repair(tmp_path) == []
    assert path.read_text(encoding="utf-8") == before


def test_prose_survives_a_polarity_flip(tmp_path):
    """`Try:` and `Note:` are the reviewer's words, not a verdict to re-derive."""
    path = _overlay(tmp_path, _SWA_LESSON)
    _card(tmp_path, "E-026", EvidenceDecision.ACCEPTED)
    _repair(tmp_path)
    text = path.read_text(encoding="utf-8")

    assert "Try: Investigate: anchor_target leakage" in text
    assert "Note: CV delta -3.8261" in text


def test_repair_drops_a_record_reference_rather_than_rewriting_it(tmp_path):
    """Caught on the real rogii overlays: an earlier version deferred this to
    the write guard and so wrote `Keep: hyp:H-005` straight back out."""
    path = _overlay(
        tmp_path,
        "<!-- lesson:E-026 -->\n## Lesson `E-026`\n- Avoid: SWA\n- Avoid: hyp:H-005\n",
    )
    _card(tmp_path, "E-026", EvidenceDecision.ACCEPTED)
    _repair(tmp_path)
    text = path.read_text(encoding="utf-8")

    assert "- Keep: SWA" in text
    assert "hyp:H-005" not in text
    assert record_references_in_overlays(tmp_path) == []


def test_a_record_reference_in_an_overlay_is_reported(tmp_path):
    _overlay(tmp_path, "<!-- lesson:E-003 -->\n## Lesson `E-003`\n- Keep: hyp:H-010\n")
    assert record_references_in_overlays(tmp_path) == ["code_engineer.md"]


def test_a_clean_overlay_reports_no_leak(tmp_path):
    _overlay(tmp_path, "<!-- lesson:E-003 -->\n## Lesson `E-003`\n- Keep: SWA\n")
    assert record_references_in_overlays(tmp_path) == []
