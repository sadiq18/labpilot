"""M20 criterion 4, the discovery half: find the views, do not list them.

Scope and what this does *not* cover are in
`docs/research-os/autonomy-roadmap/15-gates-must-fail.md`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers.baseline_campaign import HEALTHY, run_baseline_campaign

from labpilot.accessor.common.derived import derived_note, strip_derived_note
from labpilot.research_engine.evidence.models import EvidenceCard, EvidenceDecision
from labpilot.research_engine.evidence.overlay_repair import repair_skill_overlays
from labpilot.research_engine.evidence.store import EvidenceCardStore
from labpilot.research_engine.shared.skills import (
    load_skill_overlay,
    overlay_dir,
    stamped_overlay,
)


def is_stamped(text: str) -> bool:
    """Whether the text opens with a provenance note."""
    return strip_derived_note(text) != text


def unexplained_views(root: Path) -> list[str]:
    """Markdown under `root` that carries no provenance stamp."""
    return [
        path.as_posix()[len(root.as_posix()) :]
        for path in sorted(root.rglob("*.md"))
        if not is_stamped(path.read_text(encoding="utf-8", errors="ignore"))
    ]


@pytest.mark.slow
def test_a_campaign_leaves_no_markdown_that_says_nothing(tmp_path, monkeypatch):
    """The rule, over whatever the campaign wrote rather than over a list."""
    execution, _ = run_baseline_campaign(tmp_path, monkeypatch, HEALTHY)
    assert execution.status == "succeeded", execution.error

    unexplained = unexplained_views(tmp_path)

    assert not unexplained, (
        f"persisted markdown with no provenance and no rebuilder: {unexplained}. "
        "Stamp it at its write site with `derived_note`, or write it where a "
        "rebuilder already rewrites it from source."
    )


@pytest.mark.slow
def test_the_walk_reaches_the_views_that_exist(tmp_path, monkeypatch):
    """Non-vacuity: an empty walk satisfies the rule for the wrong reason."""
    run_baseline_campaign(tmp_path, monkeypatch, HEALTHY)

    names = {path.name for path in tmp_path.rglob("*.md")}

    assert {"profile.md", "P-001.md", "E-001_report.md", "report.md"} <= names
    # `overlay_dir()` builds a path and never touches disk, so `any(overlay_dir(...))`
    # was true even with the campaign's overlay wiring deleted.
    overlays = [
        path for root in tmp_path.glob("competitions/*") for path in overlay_dir(root).glob("*.md")
    ]
    assert overlays, "the campaign wrote no skill overlays"


def test_a_view_added_without_a_stamp_is_reported(tmp_path):
    """The check itself. The campaign proves today's writers comply; this proves
    tomorrow's non-compliant one is caught."""
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "summary.md").write_text("# Summary\n\nfrom the DB\n")

    assert unexplained_views(tmp_path) == ["/nested/summary.md"]


def test_a_stamped_view_is_accepted(tmp_path):
    """So the check cannot pass by rejecting everything."""
    note = derived_note(source_of_record="x.json", warning="read the json")
    (tmp_path / "stamped.md").write_text(note + "\n\n# Title\n")

    assert unexplained_views(tmp_path) == []


def test_an_overlay_keeps_its_stamp_when_the_repair_pass_rewrites_it(tmp_path):
    """Overlays are rebuilt from the cards *in part* — `Try:` and `Note:` lines
    are prose the repair deliberately preserves — so they are stamped like any
    other view. The repair rebuilds a file from its blocks, and a stamp is not a
    block, so it has to be re-applied or the next walk reports the file.
    """
    competition = "demo"
    knowledge = tmp_path / "knowledge"
    overlays = overlay_dir(tmp_path)
    overlays.mkdir(parents=True)
    stale = overlays / "code_engineer.md"
    stale.write_text(
        stamped_overlay("<!-- lesson:E-001 -->\n## Lesson `E-001`\n- Avoid: SWA\n"),
        encoding="utf-8",
    )
    EvidenceCardStore(knowledge, competition).save(
        EvidenceCard(
            competition=competition,
            treatment_experiment="E-001",
            decision=EvidenceDecision.ACCEPTED,
        )
    )

    changed = repair_skill_overlays(tmp_path, knowledge, competition)

    assert changed == ["code_engineer.md"]
    text = stale.read_text(encoding="utf-8")
    assert "Avoid: SWA" not in text
    assert is_stamped(text), "the repair dropped the stamp; the next walk would report it"
    assert unexplained_views(tmp_path) == []


def test_the_prompt_reader_strips_the_note(tmp_path):
    """The fifth reader, and the one that reaches six agents' system prompts.

    Reverting it to a plain read left the whole suite green: the ~250-character
    note then heads the overlay chunk of every prompt, inside a 1800-character
    budget.
    """
    overlays = overlay_dir(tmp_path)
    overlays.mkdir(parents=True)
    (overlays / "code_engineer.md").write_text(stamped_overlay("- Keep: SWA\n"), encoding="utf-8")

    injected = load_skill_overlay(tmp_path, "code_engineer")

    assert injected == "- Keep: SWA"


def test_an_overlay_written_before_stamping_is_migrated_by_the_repair(tmp_path):
    """Repair only wrote when content changed, so a legacy overlay that already
    agreed with the cards kept neither of criterion 4's options forever."""
    competition = "demo"
    knowledge = tmp_path / "knowledge"
    overlays = overlay_dir(tmp_path)
    overlays.mkdir(parents=True)
    legacy = overlays / "code_engineer.md"
    legacy.write_text("<!-- lesson:E-026 -->\n## Lesson `E-026`\n- Keep: SWA\n", encoding="utf-8")
    EvidenceCardStore(knowledge, competition).save(
        EvidenceCard(
            competition=competition,
            treatment_experiment="E-026",
            decision=EvidenceDecision.ACCEPTED,
        )
    )

    repair_skill_overlays(tmp_path, knowledge, competition)

    text = legacy.read_text(encoding="utf-8")
    assert is_stamped(text)
    assert "- Keep: SWA" in text
    assert unexplained_views(tmp_path) == []


def test_an_emptied_overlay_is_migrated_too(tmp_path):
    """The previous repair wrote an empty file when every lesson was dropped, so
    pre-branch workspaces carry zero-byte overlays. The migration skipped them:
    the empty check ran before it."""
    competition = "demo"
    knowledge = tmp_path / "knowledge"
    overlays = overlay_dir(tmp_path)
    overlays.mkdir(parents=True)
    emptied = overlays / "code_engineer.md"
    emptied.write_text("", encoding="utf-8")
    EvidenceCardStore(knowledge, competition).save(
        EvidenceCard(
            competition=competition,
            treatment_experiment="E-001",
            decision=EvidenceDecision.ACCEPTED,
        )
    )

    repair_skill_overlays(tmp_path, knowledge, competition)

    assert is_stamped(emptied.read_text(encoding="utf-8"))
    assert unexplained_views(tmp_path) == []


def test_the_overlay_stamp_does_not_point_at_a_sibling_that_is_not_there(tmp_path):
    """Overlays live under the competition workspace and the cards under the
    knowledge directory, so a bare `research/evidence/` resolves to nothing from
    where the file sits — the third stamp on this milestone to name the wrong
    place."""
    overlays = overlay_dir(tmp_path)
    overlays.mkdir(parents=True)
    written = overlays / "code_engineer.md"
    written.write_text(stamped_overlay("- Keep: SWA\n"), encoding="utf-8")

    note = written.read_text(encoding="utf-8").splitlines()[0]

    assert "<knowledge>" in note, "the stamp must say which tree the cards are in"


def test_a_second_lesson_does_not_add_a_second_stamp(tmp_path):
    """The strip on the *write* path. Without it each upsert prepends another
    note, and `load_skill_overlay` strips only the leading one — so the rest
    reach the prompt."""
    from labpilot.research_engine.shared.skills import upsert_skill_overlay

    for lesson in ("E-001", "E-002"):
        upsert_skill_overlay(tmp_path, "code_engineer", lesson_id=lesson, keep=["SWA"])

    written = (overlay_dir(tmp_path) / "code_engineer.md").read_text(encoding="utf-8")

    assert written.count("Derived view") == 1
    assert "Derived view" not in load_skill_overlay(tmp_path, "code_engineer")


def test_an_unwritable_overlay_does_not_abort_the_repair(tmp_path, monkeypatch):
    """`repair must never break a run` — the sibling write is guarded and the
    stamp-only branch was not, so one read-only file skipped every overlay after
    it and the passes that follow."""
    competition = "demo"
    knowledge = tmp_path / "knowledge"
    overlays = overlay_dir(tmp_path)
    overlays.mkdir(parents=True)
    blocked = overlays / "aaa_agent.md"
    blocked.write_text("", encoding="utf-8")
    later = overlays / "zzz_agent.md"
    later.write_text("<!-- lesson:E-001 -->\n## Lesson `E-001`\n- Keep: SWA\n", encoding="utf-8")
    EvidenceCardStore(knowledge, competition).save(
        EvidenceCard(
            competition=competition,
            treatment_experiment="E-001",
            decision=EvidenceDecision.ACCEPTED,
        )
    )

    # Patched rather than `chmod(0o444)`: root ignores the permission bit, so on
    # a root-running runner the write would succeed and this would pass without
    # reaching the handler at all.
    real_write = Path.write_text

    def refuse_the_blocked_one(self, data, *args, **kwargs):
        if self == blocked:
            raise OSError("read-only file system")
        return real_write(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", refuse_the_blocked_one)
    repair_skill_overlays(tmp_path, knowledge, competition)
    monkeypatch.undo()

    assert is_stamped(later.read_text(encoding="utf-8")), "the file after it was skipped"


def test_the_on_disk_budget_bounds_the_file_including_its_stamp(tmp_path):
    """`ON_DISK_CHAR_BUDGET` names the on-disk overlay, so the note comes out of
    it. Summarising the body alone and then prepending ~240 characters left the
    file over the budget it is measured against."""
    from labpilot.research_engine.shared.skills import overlay_note_cost, upsert_skill_overlay

    budget = overlay_note_cost() + 200
    for lesson in range(30):
        upsert_skill_overlay(
            tmp_path,
            "code_engineer",
            lesson_id=f"E-{lesson:03d}",
            keep=["a technique with a reasonably long name"],
            on_disk_budget=budget,
        )

    written = (overlay_dir(tmp_path) / "code_engineer.md").read_text(encoding="utf-8")

    assert len(written) <= budget, f"{len(written)} chars against a {budget} budget"
    assert is_stamped(written), "the note must survive summarisation, not be truncated by it"
