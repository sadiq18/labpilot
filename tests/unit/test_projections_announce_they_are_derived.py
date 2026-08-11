"""A derived file must not be mistakable for the source of record.

Two wrong diagnoses came from reading one, and both were *reasonable* reads —
the files agreed with each other and carried nothing marking them derived:

* `evidence-log.md` correction #1: *"All 17 plans are stuck at `ready`; nothing
  writes `done`"* — read off `research/plans/*.json`. Measured on rogii
  2026-08-08 the DB held `done=16, abandoned=3` while **19 of 19** projections
  said `ready`.
* 2026-08-08: "the evidence cards are inverted, the knowledge base is
  untrustworthy" — read off `artifacts/evidence_card_*.json`, while the store
  under `research/evidence/` had already been repaired.

Neither file is *wrong* for what it is. Both are snapshots that stopped
tracking. The property under test is that saying so is impossible to miss.
"""

from __future__ import annotations

import json

from labpilot.research_engine.planner.schemas.models import ResearchPlan
from labpilot.research_engine.planner.serializer import (
    PROJECTION_KEY,
    plan_to_json,
    render_markdown,
)


def _plan(**kw) -> ResearchPlan:
    now = "2026-08-08T00:00:00+00:00"
    return ResearchPlan(
        id=kw.pop("id", "P-001"),
        competition=kw.pop("competition", "rogii-wellbore-geology-prediction"),
        goal=kw.pop("goal", "reduce MSE"),
        hypothesis_id=kw.pop("hypothesis_id", "H-001"),
        created_at=kw.pop("created_at", now),
        updated_at=kw.pop("updated_at", now),
        **kw,
    )


def test_the_json_projection_says_it_is_not_authoritative():
    payload = json.loads(plan_to_json(_plan()))

    stamp = payload[PROJECTION_KEY]
    assert stamp["authoritative"] is False
    assert "research_plans" in stamp["source_of_record"]
    assert stamp["generated_at"]


def test_the_stamp_names_status_as_the_field_that_goes_stale():
    """Status is the field that actually diverged, so it is named explicitly.

    A generic "may be out of date" would not have stopped either misdiagnosis —
    both were *about status*.
    """
    stamp = json.loads(plan_to_json(_plan()))[PROJECTION_KEY]
    warning = stamp["warning"].lower()

    assert "status" in warning
    assert "research_plans" in warning


def test_the_plan_itself_still_round_trips():
    """The stamp must not cost the artifact its purpose: diff and inspect."""
    plan = _plan(id="P-042", goal="ensemble lgb with cb")
    payload = json.loads(plan_to_json(plan))

    assert payload["id"] == "P-042"
    assert payload["goal"] == "ensemble lgb with cb"
    restored = ResearchPlan.model_validate(
        {k: v for k, v in payload.items() if k != PROJECTION_KEY}
    )
    assert restored.id == plan.id
    assert restored.goal == plan.goal


def test_the_stamp_sorts_before_the_plan_fields():
    """A reader who opens the file and reads one line should see the warning."""
    first_key = next(iter(json.loads(plan_to_json(_plan()))))
    assert first_key == PROJECTION_KEY


def test_the_markdown_warns_above_the_status_line(tmp_path):
    """Markdown is what a human actually opens, so order matters here too.

    Asserted on the **written file**, not on `render_markdown`. The stamp moved
    to `write_projections` because the renderer is also used by `plan show
    --format markdown`, which reads the plan live from the DB — there the
    warning that the status is stale is simply false. The guarantee this test
    exists for is about the file that persists, and it still holds.
    """
    from labpilot.research_engine.planner.serializer import write_projections

    _, md_path = write_projections(
        _plan(), knowledge_dir=tmp_path / "knowledge", competition="demo"
    )
    text = md_path.read_text(encoding="utf-8")

    warning_at = text.index("source of record")
    status_at = text.index("**Status (at creation):**")
    assert warning_at < status_at


def test_a_plan_rendered_live_is_not_told_it_is_stale():
    """`plan show --format markdown` renders straight from `PlanStore`, so the
    reading is current. Announcing "not refreshed since" on a live read trains a
    reader to discount a fact that is true — the opposite of what the stamp is
    for, and the reason it belongs to the writer rather than the renderer."""
    text = render_markdown(_plan())

    assert "not authoritative" not in text.lower()
    assert "source of record" not in text


def test_the_markdown_does_not_present_status_as_current():
    """`- **Status:** ready` reads as fact. It is a creation-time value."""
    text = render_markdown(_plan())

    assert "**Status (at creation):**" in text
    assert "- **Status:** " not in text
