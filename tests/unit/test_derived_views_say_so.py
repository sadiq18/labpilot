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

Enforcing it found three more, already shipped and all unstamped:

* `comparison.md`, beside `comparison.json` — the writer's own docstring calls
  that *"(source of truth)"* and this one *"(view)"*.
* `profile.md`, beside the `profile.json` the same call writes.
* `research_brief.md`, rendered from `analyze.json` and *not written with it*:
  `research analyze --skip-hypothesize` rewrites the JSON and skips the brief.
  The only one of the four read back as **machine** input, by the planner under a
  2000-character budget — hence `strip_derived_note`.

**The stamp belongs to the writer, not the renderer.** The first version put it
in `render_markdown`, and two callers render *live* rather than persisting:
`experiments compare --format markdown` recomputes whenever the stored JSON
records a different pair, and `plan show --format markdown` reads the DB
directly. Both were then told to "read the JSON" — for a file that may not exist,
or that describes a different comparison. A stamp that misdirects is worse than
none, and it is the exact failure the stamp exists to prevent. Moving it to the
write sites also fixed that for plan projections, where the stale warning had
been printed on live reads since before this branch.

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
import pathlib

from labpilot.accessor.common.derived import (
    derived_note,
    derived_stamp,
    strip_derived_note,
)


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
    # The source of truth is not a view and must not claim to be one. Asserted
    # by round-tripping it, because a pydantic dump of a fixed model can never
    # contain a stray key — the first version checked exactly that and could not
    # fail.
    source = (tmp_path / "comparison.json").read_text(encoding="utf-8")
    assert not source.lstrip().startswith(">")
    assert "not authoritative" not in json.loads(source).get("verdict_reason", "")


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


def _persisted_views(root) -> list[tuple[str, str]]:
    """Every markdown view the system *writes*, named and read back off disk.

    Written, not rendered. The rule is about a file that outlives its source, so
    the check has to be on the file — and putting it on the renderer got this
    wrong in both directions: it stamped two callers that render live and would
    have missed a stamp applied at the write site.
    """
    from labpilot.accessor.profiler.report import write_profile
    from labpilot.accessor.profiler.tabular import DatasetProfile
    from labpilot.research_engine.intelligence.brief.models import ResearchBrief
    from labpilot.research_engine.intelligence.renderers.markdown import write_brief
    from labpilot.research_engine.planner.serializer import write_projections
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

    written: list[tuple[str, str]] = []

    comparison_dir = root / "run"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    write_comparison(comparison_dir, comparison)
    written.append(("comparison.md", (comparison_dir / "comparison.md").read_text()))

    profile_dir = root / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    write_profile(profile_dir, DatasetProfile(competition="demo"))
    written.append(("profile.md", (profile_dir / "profile.md").read_text()))

    _, plan_md = write_projections(_plan(), knowledge_dir=root / "knowledge", competition="demo")
    written.append(("plan projection", plan_md.read_text()))

    brief_path = write_brief(ResearchBrief(competition="demo"), root / "research_brief.md")
    written.append(("research_brief.md", brief_path.read_text()))

    # The fifth, driven through its capability because that is the only writer.
    # It lands in `reports_dir`, which `WorkspaceProvider` rglobs into LLM
    # context, so an unstamped copy travels further than the others.
    from helpers.capability_context import capability_context

    from labpilot.research_engine.execution.capabilities.reporting import ReportingCapability
    from labpilot.research_engine.planner.schemas.task_types import TaskType

    report_root = root / "report_ws"
    context = capability_context(report_root, task_type=TaskType.GENERATE_REPORT)
    (context.workspace_root / "metrics.json").write_text('{"cv": 0.8}', encoding="utf-8")
    result = ReportingCapability().execute(context)
    report = next(p for p in result.paths if p.endswith("_report.md"))
    written.append(("execution report", pathlib.Path(report).read_text(encoding="utf-8")))

    return written


def test_every_persisted_markdown_view_says_it_is_one(tmp_path):
    """The rule. Each view is written by its real writer and read back, so a
    stamp that is never applied at the write site cannot satisfy it.

    The count is not restated anywhere: an earlier version parametrised over
    `range(3)` *and* asserted `len(...) == 3` separately, so adding a fourth
    renderer failed only the count, and the natural fix left the new one
    unchecked.
    """
    views = _persisted_views(tmp_path)

    assert views, "no views were produced — has a writer moved?"
    for name, text in views:
        first = text.lstrip().splitlines()[0]
        # The first line, not merely somewhere in the file: setting a view's own
        # content to "not authoritative; read comparison.json" and deleting the
        # stamp turned the substring version of this green.
        assert first.startswith(">"), f"{name}: the stamp is not the first thing read"
        assert "not authoritative" in first.lower(), f"{name} does not say it is a view"


def test_a_machine_reader_can_drop_the_block_it_does_not_need(tmp_path):
    """`research_brief.md` is read by the planner under a 2000-character budget,
    where the stamp is 200 characters that displace the brief. Stripping is what
    lets the file carry one at all."""
    views = dict(_persisted_views(tmp_path))
    stamped = views["research_brief.md"]

    stripped = strip_derived_note(stamped)

    assert not stripped.lstrip().startswith(">")
    assert "not authoritative" not in stripped.lower()
    assert stripped.strip(), "stripping must not empty the file"
    # Idempotent, and inert on text that was never stamped.
    assert strip_derived_note(stripped) == stripped


def test_stripping_keeps_a_quote_that_belongs_to_the_content():
    """Only the leading provenance block goes. A view whose own body opens with a
    blockquote after the stamp keeps it."""
    body = "# Title\n\n> a quotation the author wrote\n"
    stamped = derived_note(source_of_record="x.json", warning="w") + "\n\n" + body

    assert strip_derived_note(stamped) == body.rstrip("\n")
    assert strip_derived_note(body) == body


def test_a_real_reader_of_a_stamped_view_gets_the_content(tmp_path):
    """Reported reviewing this branch, and mutation-proven: replacing
    `read_derived`'s body with a plain `read_text` left the whole suite green.

    Nothing exercised it. Every fixture that reached a call site wrote an
    *unstamped* brief, so the strip path was never entered — the coverage was of
    the helper in isolation and of call sites that had nothing to strip. The
    regression it guards is the one round 2 fixed: 277 characters of "distrust
    this" at the head of the codegen prompt's research window.

    Driven through `_brief_excerpt`, a real call site, on a file written by the
    real writer.
    """
    from labpilot.research_engine.intelligence.brief.models import ResearchBrief
    from labpilot.research_engine.intelligence.paths import ResearchPaths
    from labpilot.research_engine.intelligence.renderers.markdown import write_brief
    from labpilot.research_engine.planner.retrieval import _brief_excerpt

    knowledge = tmp_path / "knowledge"
    paths = ResearchPaths(knowledge, "demo")
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    write_brief(ResearchBrief(competition="demo"), paths.brief_path)

    on_disk = paths.brief_path.read_text(encoding="utf-8")
    excerpt = _brief_excerpt(knowledge, "demo")

    assert on_disk.lstrip().startswith(">"), "the fixture must actually be stamped"
    assert "not authoritative" not in excerpt.lower()
    assert not excerpt.lstrip().startswith(">")
    assert excerpt.strip(), "stripping must not empty what the planner sees"


def test_read_derived_leaves_a_file_that_was_never_stamped_alone(tmp_path):
    """The other direction, so the fix cannot be "strip the first paragraph"."""
    from labpilot.accessor.common.derived import read_derived

    plain = tmp_path / "notes.md"
    plain.write_text("# Notes\n\n> a quotation\n", encoding="utf-8")

    assert read_derived(plain) == "# Notes\n\n> a quotation\n"
