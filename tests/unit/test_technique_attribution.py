"""A hypothesis ID must never be attributed as a technique.

Measured on the rogii workspace before this guard: `hyp:H-010` was the single
most common "technique" in the knowledge base — 11 durable records — ahead of
every real one. The loop that produced it:

    outcome.py appends `hyp:{hypothesis_id}` to an experiment's tags
      -> evidence attribution falls back to tags when no technique is set
      -> a belief is written claiming `hyp:H-010` is a technique
      -> hypothesis generation reads that belief
      -> the planner writes it to plan.metadata["technique"]
      -> codegen is asked to implement "hyp:H-010"

Nothing downstream can recover from this: a technique registry cannot resolve a
hypothesis ID, and a frontier model cannot implement one either. So the guard
belongs at the point of attribution, which is where `fork:` was already handled
and `hyp:` was not.
"""

from __future__ import annotations

import pytest

from labpilot.research_engine.evidence.builder import _reusable_for
from labpilot.research_engine.shared.labels import is_record_reference


@pytest.mark.parametrize(
    "tag",
    ["hyp:H-010", "HYP:H-010", "hyp:H-BASELINE", "fork:H-010", "  fork:H-3  "],
)
def test_record_references_are_not_techniques(tag):
    assert is_record_reference(tag) is True


@pytest.mark.parametrize(
    "tag",
    ["target_encoding", "lag_features", "rolling_features", "catboost", "hypothesis"],
)
def test_real_technique_names_survive(tag):
    """`hypothesis` must survive — the guard keys on the `hyp:` prefix, not on
    a substring, so a real technique that merely starts with those letters is
    not swallowed."""
    assert is_record_reference(tag) is False


def test_reusable_tags_drop_record_references():
    """The exact tag list observed on rogii's P-009."""
    tags = _reusable_for(
        "rogii-wellbore-geology-prediction",
        {
            "change_category": "other",
            "tags": ["hyp:H-BASELINE", "stacked", "improvement", "fork:H-010"],
        },
    )
    assert not [t for t in tags if is_record_reference(t)], (
        f"record references leaked into reuse tags: {tags}"
    )


def test_attribution_prefers_a_real_technique_over_tags():
    """The tag fallback only applies when no technique is declared, so a real
    name must never be displaced by bookkeeping tags."""
    tags = _reusable_for(
        "rogii-wellbore-geology-prediction",
        {"change_category": "features", "tags": ["target_encoding", "hyp:H-010"]},
    )
    assert "target_encoding" in tags
    assert "hyp:H-010" not in tags


def _card_for(tmp_path, *, tags, technique=""):
    """Build a real evidence card, exercising the attribution fallback."""
    from labpilot.research_engine.evidence.builder import build_evidence_card
    from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore

    competition = "attribution-demo"
    hyp = HypothesisStore(tmp_path, competition).create(
        observation="o",
        reason="r",
        prediction="p",
        confidence=0.7,
        expected_impact=0.004,
        tags=tags,
        technique=technique,
        parent_hypothesis_id="H-010",
    )
    return build_evidence_card(
        knowledge_dir=tmp_path,
        competition=competition,
        treatment_execution_id="E-014",
        treatment_metrics={"cv_accuracy": 0.91, "cv_std": 0.01, "train_time_s": 100.0},
        control_execution_id="E-008",
        control_metrics={"cv_accuracy": 0.90, "cv_std": 0.02, "train_time_s": 80.0},
        hypothesis_id=hyp.id,
        plan_id="P-004",
        plan_metadata={"change_category": "other", "tags": tags},
        maximize=True,  # accuracy: higher is better
        persist=False,
    )


def test_no_hypothesis_id_is_attributed_as_a_technique(tmp_path):
    """The measured failure, end to end.

    rogii's P-004..P-008 all carried `technique=''` on the hypothesis and
    `hyp:H-010` in tags, so attribution fell through to the tags and wrote
    `hyp:H-010` into a durable belief — 11 times.
    """
    card = _card_for(tmp_path, tags=["hyp:H-010", "stacked", "improvement", "fork:H-010"])

    # Deliberately a literal prefix check rather than `_is_record_reference`:
    # the assertion must not be expressed in terms of the helper it is testing,
    # or reverting the fix would break collection instead of failing the test.
    attributed = list(card.technique_attribution)
    leaked = [t for t in attributed if t.lower().startswith(("hyp:", "fork:"))]
    assert not leaked, f"a record reference was attributed as a technique: {attributed}"


def test_a_real_technique_in_tags_is_still_attributed(tmp_path):
    """The control: the fallback must keep working for genuine names, or the
    fix above would be indistinguishable from disabling attribution."""
    card = _card_for(tmp_path, tags=["hyp:H-010", "rolling_features", "stacked"])
    assert "rolling_features" in card.technique_attribution


# --- the ledger guard that looked like protection and never fired -----------


def test_normalising_before_the_prefix_check_defeats_it():
    """Why `shared/labels.py` insists on the raw label.

    `_index_technique` tested `normalize_label(name).startswith("hyp:")`.
    Normalisation strips non-alphanumerics, so the colon the prefix depends on
    is gone before the comparison — the condition could never be true. Five
    `hyp:*` rows reached `techniques.name` through that gap.
    """
    from labpilot.research_engine.intelligence.retrieval.fetchers import normalize_label

    assert normalize_label("hyp:H-010") == "hyph010"
    assert normalize_label("hyp:H-010").startswith("hyp:") is False, (
        "the old guard's condition — it cannot fire"
    )
    assert is_record_reference("hyp:H-010") is True, "the rule must see the raw label"


def test_index_technique_drops_record_references():
    """The guard's effect, asserted on the index it populates."""
    from labpilot.research_engine.intelligence.hypothesis.ledger import _index_technique

    index: dict = {}
    for name in ("hyp:H-010", "fork:H-003", "rolling_features", "target_encoding"):
        _index_technique(index, name)

    names = sorted(record.name for record in index.values())
    assert names == ["rolling_features", "target_encoding"]
