"""A record reference is never a technique — now enforced at the writer.

`shared/labels.py` documents two prior attempts at this rule and why each
failed. This is the third site with the same omission: `_techniques_from_plan`
filtered `fork:` and let `hyp:` through, exactly as `evidence/builder.py` once
did. Meanwhile `merge_technique` — the single door into the vocabulary — called
no guard at all, so filtering readers could never shrink a table that kept
being written to. Measured on rogii: 13 record ids in `techniques.name`, up
from the 5 that `labels.py` recorded.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from labpilot.research_engine.execution.outcome import _techniques_from_plan
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.planner.schemas.models import ResearchPlan


def _plan(**metadata) -> ResearchPlan:
    return ResearchPlan(
        id="P-001",
        competition="rogii",
        goal="beat the baseline",
        hypothesis_id=metadata.pop("hypothesis_id", "H-010"),
        metadata=metadata,
        created_at=datetime.now(UTC).isoformat(),
        updated_at=datetime.now(UTC).isoformat(),
    )


# --- the source: outcome.py appends `hyp:{id}` for provenance ---------------


def test_the_hypothesis_id_it_appends_is_not_returned_as_a_technique():
    """Line 458 appends `hyp:{hypothesis_id}`, so this function is the source.
    Six rogii plans asked codegen to implement `hyp:H-010`."""
    assert "hyp:H-010" not in _techniques_from_plan(_plan(technique="SWA"))


def test_fork_references_are_still_filtered():
    """The old check caught `fork:` only; it must keep catching it."""
    out = _techniques_from_plan(_plan(tags=["fork:H-003", "SWA"]))
    assert "fork:H-003" not in out


def test_real_techniques_still_survive():
    """A filter that drops everything is not a filter."""
    out = _techniques_from_plan(_plan(technique="SWA", technique_stack=["vit", "SWA"]))
    assert "SWA" in out
    assert "vit" in out


def test_a_record_reference_inside_the_stack_is_dropped():
    out = _techniques_from_plan(_plan(technique_stack=["vit", "hyp:H-BASELINE"]))
    assert out == ["vit"]


# --- the chokepoint: the single door into the vocabulary --------------------


def test_merge_technique_refuses_a_record_reference(tmp_path: Path):
    """Loud, not a silent skip. `labels.py` records that the previous guard
    failed by reading as protection while doing nothing; returning an empty id
    would repeat that and hand `link_artifact_technique` a dangling row."""
    with KnowledgeStore(tmp_path / "knowledge", "rogii") as store:
        with pytest.raises(ValueError, match="record reference"):
            store.merge_technique("hyp:H-010")
        with pytest.raises(ValueError, match="record reference"):
            store.merge_technique("fork:H-003")


def test_merge_technique_still_accepts_a_real_name(tmp_path: Path):
    with KnowledgeStore(tmp_path / "knowledge", "rogii") as store:
        assert store.merge_technique("SWA")


def test_the_vocabulary_stays_clean_after_a_refused_write(tmp_path: Path):
    """The point of guarding the writer rather than the readers: the row must
    never exist, so no later reader has to remember to filter it."""
    with KnowledgeStore(tmp_path / "knowledge", "rogii") as store:
        store.merge_technique("SWA")
        with pytest.raises(ValueError):
            store.merge_technique("hyp:H-010")
        names = {
            row["name"] for row in store._conn.execute("SELECT name FROM techniques").fetchall()
        }
    assert names == {"SWA"}
