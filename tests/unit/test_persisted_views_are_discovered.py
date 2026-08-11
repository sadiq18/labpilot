"""M20 criterion 4, the discovery half: find the views, do not list them.

Scope and what this does *not* cover are in
`docs/research-os/autonomy-roadmap/15-gates-must-fail.md`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers.campaign import HEALTHY, run_baseline_campaign

from labpilot.accessor.common.derived import derived_note, strip_derived_note
from labpilot.research_engine.evidence.models import EvidenceCard, EvidenceDecision
from labpilot.research_engine.evidence.overlay_repair import repair_skill_overlays
from labpilot.research_engine.evidence.store import EvidenceCardStore
from labpilot.research_engine.shared.skills import overlay_dir


def is_stamped(text: str) -> bool:
    """Whether the text opens with a provenance note."""
    return strip_derived_note(text) != text


def rederived_dirs(root: Path) -> set[Path]:
    """Directories whose markdown is rebuilt rather than stamped.

    Resolved through `overlay_dir`, production's own answer for where overlays
    live, so the exemption cannot drift wider than the thing it names.
    """
    found = set()
    for competition_root in root.glob("competitions/*"):
        directory = overlay_dir(competition_root)
        if directory is not None:
            found.add(directory.resolve())
    return found


def unexplained_views(root: Path) -> list[str]:
    """Markdown under `root` that neither carries a stamp nor is rebuilt."""
    exempt = rederived_dirs(root)
    return [
        path.as_posix()[len(root.as_posix()) :]
        for path in sorted(root.rglob("*.md"))
        if path.parent.resolve() not in exempt
        and not is_stamped(path.read_text(encoding="utf-8", errors="ignore"))
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
    assert any(path.parent.resolve() in rederived_dirs(tmp_path) for path in tmp_path.rglob("*.md"))


def test_a_view_added_without_a_stamp_is_reported(tmp_path):
    """The check itself. The campaign proves today's writers comply; this proves
    tomorrow's non-compliant one is caught."""
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "summary.md").write_text("# Summary\n\nfrom the DB\n")

    assert unexplained_views(tmp_path) == ["/nested/summary.md"]


def test_a_stamped_view_and_a_rebuilt_one_are_both_accepted(tmp_path):
    """Both of criterion 4's options, so the check cannot pass by rejecting one."""
    note = derived_note(source_of_record="x.json", warning="read the json")
    (tmp_path / "stamped.md").write_text(note + "\n\n# Title\n")
    overlays = overlay_dir(tmp_path / "competitions" / "demo")
    overlays.mkdir(parents=True)
    (overlays / "agent.md").write_text("- Keep: nothing\n")

    assert unexplained_views(tmp_path) == []


def test_only_the_overlay_directory_itself_is_exempt(tmp_path):
    """A neighbour of the exempt directory is not exempt. Substring matching on
    the path made `skills_v2/`, `skills-archive/` and nested files exempt too,
    while the rebuilder only ever rewrites `<overlays>/*.md`."""
    competition = tmp_path / "competitions" / "demo"
    exempt = overlay_dir(competition)
    exempt.mkdir(parents=True)
    (exempt / "agent.md").write_text("- Keep: nothing\n")
    for neighbour in ("skills_v2", "skills-archive"):
        directory = exempt.parent / neighbour
        directory.mkdir()
        (directory / "view.md").write_text("# Archived\n")
    nested = exempt / "deep"
    nested.mkdir()
    (nested / "view.md").write_text("# Nested\n")

    reported = unexplained_views(tmp_path)

    assert len(reported) == 3
    assert all("agent.md" not in entry for entry in reported)


def test_the_exempt_directory_is_rebuilt_from_its_source(tmp_path):
    """The exemption, earned rather than named.

    Asserting the rebuilder merely *exists* passed for any module attribute —
    `logger` satisfied it. This corrupts an overlay and requires the named
    rebuilder to correct it from the cards.
    """
    competition = "demo"
    knowledge = tmp_path / "knowledge"
    overlays = overlay_dir(tmp_path)
    overlays.mkdir(parents=True)
    stale = overlays / "code_engineer.md"
    stale.write_text("<!-- lesson:E-001 -->\n## Lesson `E-001`\n- Avoid: SWA\n", encoding="utf-8")
    EvidenceCardStore(knowledge, competition).save(
        EvidenceCard(
            competition=competition,
            treatment_experiment="E-001",
            decision=EvidenceDecision.ACCEPTED,
        )
    )

    changed = repair_skill_overlays(tmp_path, knowledge, competition)

    assert changed == ["code_engineer.md"]
    assert "Avoid: SWA" not in stale.read_text(encoding="utf-8")
