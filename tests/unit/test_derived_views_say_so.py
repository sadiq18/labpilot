"""M20 exit criterion 4: a derived artifact re-derives, or says it is derived.

Three defects were one shape — a file written once from a source that later
changed, silently disagreeing with it. Plan projections said `ready` for all 19
plans while the DB said `done=16, abandoned=3`. Overlays told six agents to avoid
`SWA`, the only technique that ever improved the metric, because the card behind
that lesson had been repaired and the overlay had no path back to it. Beliefs
recorded the same technique `negative` after all 15 cards were re-oriented and
zero beliefs moved. Two wrong diagnoses were read off these files six days apart,
by the same author, and both were reasonable: every file agreed with every other.

Each of those four got its own answer — `repair_card_directions`,
`rederive_beliefs_from_cards`, `repair_skill_overlays`, and a staleness stamp on
projections — and each got its own test. What none of them got was a rule, so the
criterion asks for one enforced over the *writers*.

Enforcing it found two more, already shipped and both unstamped:

* `comparison.md`, beside `comparison.json` — the writer's own docstring calls
  that *"(source of truth)"* and this one *"(view)"*.
* `profile.md`, beside the `profile.json` the same call writes.

`comparison.md` is the one that matters most: evidence-card directions are
repaired by a pass that runs every campaign, and a verdict rendered into markdown
before that repair keeps the reading the repair exists to correct.

A third candidate, `JournalProjector.render_markdown`, turned out to be compliant
already — `cli/reflect.py` prints it and nothing writes it to disk, so it takes
the criterion's *other* option and re-derives on every read. Worth recording
because "it renders markdown from a source" is not the test; "a file persists
after its source moves" is.

The stamp is one helper rather than a copy per writer, because two implementations
of one idea drifting apart is the defect criterion 2 is named after.
"""

from __future__ import annotations

import json

import pytest

from labpilot.accessor.common.derived import DERIVED_KEY, derived_note, derived_stamp


def test_a_stamp_states_it_is_not_authoritative_and_names_its_source():
    """The three facts a reader needs: that this is a copy, what it is a copy of,
    and when it was taken. A stamp missing the source is a warning nobody can
    act on — the misdiagnoses this criterion is about were resolved by going to
    the source of record, and the point of the stamp is to say where that is."""
    stamp = derived_stamp(source_of_record="comparison.json", warning="do not edit")

    assert stamp["authoritative"] is False
    assert stamp["source_of_record"] == "comparison.json"
    assert stamp["warning"] == "do not edit"
    assert isinstance(stamp["generated_at"], str) and stamp["generated_at"]


def test_the_markdown_note_carries_the_same_three_facts():
    """A markdown view cannot hold a JSON key, and a reader of one is exactly the
    reader who was misled — the plan projections that produced the first wrong
    diagnosis were read as markdown."""
    note = derived_note(source_of_record="comparison.json", warning="do not edit")

    assert note.startswith(">"), "a blockquote, so it survives rendering as a callout"
    assert "comparison.json" in note
    assert "do not edit" in note
    assert "not authoritative" in note.lower()


def test_the_plan_projection_stamp_is_built_from_the_shared_one():
    """`projection_stamp` predates the helper and keeps its own `_projection`
    key, which is already on disk in every workspace and asserted elsewhere. What
    it must not keep is its own *implementation*: a second copy of the stamp is
    how the next field gets added to one and not the other."""
    from labpilot.research_engine.planner.serializer import PROJECTION_KEY, projection_stamp

    stamp = projection_stamp()

    assert PROJECTION_KEY == "_projection", "the on-disk key must not move"
    assert set(stamp) == set(derived_stamp(source_of_record="x", warning="y"))
    assert stamp["authoritative"] is False
    assert "knowledge.db" in str(stamp["source_of_record"])


def test_a_comparison_markdown_says_it_is_a_view_of_the_json_beside_it(tmp_path):
    """The most damaging of the three, because its source is actively repaired.

    `repair_card_directions` re-orients evidence cards every campaign — that is
    the pass that turned EV-012 from a rejection into an acceptance. A verdict
    already rendered into `comparison.md` keeps the reading the repair exists to
    correct, sitting next to a `comparison.json` that now disagrees with it.
    """
    from labpilot.research_engine.shared.experiments.comparator import write_comparison
    from labpilot.research_engine.shared.experiments.models import (
        ExperimentComparison,
        Verdict,
    )

    comparison = ExperimentComparison(
        base_id="E-001",
        compare_id="E-002",
        primary_metric_key="cv",
        metric_deltas={"cv": 0.04},
        changes=[],
        runtime_delta_seconds=None,
        runtime_delta_pct=None,
        verdict=Verdict.WORTH_KEEPING,
        verdict_reason="cv 0.80 -> 0.84",
    )

    write_comparison(tmp_path, comparison)

    view = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    # The *first* line, not merely somewhere in the file. Substring assertions
    # here are satisfied by the view's own content: setting `verdict_reason` to
    # "not authoritative; read comparison.json" and deleting the stamp entirely
    # turned this test green. Found by mutating the fixture rather than the code.
    first = view.lstrip().splitlines()[0]
    assert first.startswith(">"), "the stamp must be the first thing read"
    assert "comparison.json" in first, "the view must name the source beside it"
    assert "not authoritative" in first.lower()
    # The source of truth is not a view and must not claim to be one.
    source = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert DERIVED_KEY not in source


def test_a_dataset_profile_markdown_says_it_is_a_view(tmp_path):
    """Read by a human deciding what the data looks like, and regenerated only
    when profiling reruns — the same write-once shape."""
    from labpilot.accessor.profiler.report import write_profile
    from labpilot.accessor.profiler.tabular import DatasetProfile

    profile = DatasetProfile(competition="demo", row_count=10, column_count=2, files=["train.csv"])

    write_profile(tmp_path, profile)

    first = next(tmp_path.rglob("*.md")).read_text(encoding="utf-8").lstrip().splitlines()[0]
    assert first.startswith(">") and "not authoritative" in first.lower()


def _plan():
    from datetime import UTC, datetime

    from labpilot.research_engine.planner.schemas.models import ResearchPlan

    now = datetime.now(UTC)
    return ResearchPlan(
        id="P-001",
        competition="demo",
        hypothesis_id="H-001",
        goal="prove a view says it is one",
        created_at=now,
        updated_at=now,
    )


def _rendered_views() -> list[tuple[str, str]]:
    """Every markdown view renderer, called for real, with its output.

    Called rather than read. Criterion 1 spent seven review rounds inside a
    parser that decided things about capabilities by reading their source, and
    the lesson was that a check on what a run *did* cannot be fooled by a
    spelling nobody anticipated. `inspect.getsource(...)` searching for
    `derived_note` would pass on a renderer that imported it and never called it,
    and fail on one that wrote the same words another way.
    """
    from labpilot.accessor.profiler.report import render_markdown as render_profile
    from labpilot.accessor.profiler.tabular import DatasetProfile
    from labpilot.research_engine.planner.serializer import render_markdown as render_plan
    from labpilot.research_engine.shared.experiments.comparator import (
        render_markdown as render_comparison,
    )
    from labpilot.research_engine.shared.experiments.models import (
        ExperimentComparison,
        Verdict,
    )

    comparison = ExperimentComparison(
        base_id="E-001",
        compare_id="E-002",
        primary_metric_key="cv",
        metric_deltas={"cv": 0.04},
        changes=[],
        runtime_delta_seconds=None,
        runtime_delta_pct=None,
        verdict=Verdict.WORTH_KEEPING,
        verdict_reason="cv 0.80 -> 0.84",
    )
    return [
        ("plan projection", render_plan(_plan())),
        ("comparison", render_comparison(comparison)),
        ("dataset profile", render_profile(DatasetProfile(competition="demo"))),
    ]


@pytest.mark.parametrize("index", range(3))
def test_every_markdown_view_says_it_is_a_view(index):
    """The rule, over the writers rather than over whichever files one test
    happened to write. A renderer that stops stamping fails here even when no
    test exercises the path that puts it on disk.

    Parametrised by index so each renderer is its own test id and a single
    regression does not read as three."""
    name, rendered = _rendered_views()[index]

    assert "not authoritative" in rendered.lower(), f"{name} does not say it is a view"
    assert rendered.lstrip().startswith(">"), f"{name}'s stamp is not the first thing read"


def test_the_view_renderers_are_all_enumerated():
    """A guard on the list above: three renderers were found by following
    `render_markdown` across the codebase, and a fourth added later is invisible
    to a hand-written list. This does not solve that — it states the count so a
    reader who adds one has something to update, and `15-gates-must-fail.md`
    records auto-discovery as the part not built."""
    assert len(_rendered_views()) == 3
