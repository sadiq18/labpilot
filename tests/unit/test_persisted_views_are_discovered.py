"""M20 criterion 4, the discovery half: find the views, do not list them.

Five writers were enumerated by hand, and the fifth was found by a reviewer
rather than by the rule — which makes the rule review, wearing a test's clothes.
This walks what a campaign actually leaves on disk instead.

The rule and what it found are in
`docs/research-os/autonomy-roadmap/15-gates-must-fail.md`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers.campaign import HEALTHY, run_baseline_campaign

from labpilot.accessor.common.derived import derived_note, strip_derived_note

#: Path fragments whose markdown takes criterion 4's *other* option: rebuilt from
#: the current source on every campaign, so a stamp would be redundant. The
#: value names the rebuilder, so a claim here can be checked.
REDERIVED = {
    ".labpilot/skills": "repair_skill_overlays",
}


def is_stamped(text: str) -> bool:
    """Whether the text opens with a provenance note."""
    return strip_derived_note(text) != text


def unexplained_views(root: Path) -> list[str]:
    """Markdown under `root` that neither carries a stamp nor is re-derived."""
    found = []
    for path in sorted(root.rglob("*.md")):
        posix = path.as_posix()
        if any(fragment in posix for fragment in REDERIVED):
            continue
        if is_stamped(path.read_text(encoding="utf-8", errors="ignore")):
            continue
        found.append(posix[len(root.as_posix()) :])
    return found


@pytest.mark.slow
def test_a_campaign_leaves_no_markdown_that_says_nothing(tmp_path, monkeypatch):
    """The rule, over whatever the campaign wrote rather than over a list."""
    run_baseline_campaign(tmp_path, monkeypatch, HEALTHY)

    unexplained = unexplained_views(tmp_path)

    assert not unexplained, (
        f"persisted markdown with no provenance and no rebuilder: {unexplained}. "
        "Stamp it at its write site with `derived_note`, or add its directory to "
        "REDERIVED naming what rebuilds it."
    )


@pytest.mark.slow
def test_the_walk_reaches_the_views_that_exist(tmp_path, monkeypatch):
    """Non-vacuity. An empty walk satisfies the rule above for the wrong reason,
    and a campaign that stopped early would produce one."""
    run_baseline_campaign(tmp_path, monkeypatch, HEALTHY)

    names = {path.name for path in tmp_path.rglob("*.md")}

    assert {"profile.md", "P-001.md", "E-001_report.md", "report.md"} <= names
    assert len([p for p in tmp_path.rglob("*.md") if ".labpilot/skills" in p.as_posix()]) >= 1


def test_a_view_added_without_a_stamp_is_reported(tmp_path):
    """The check itself, which is what the rule rests on.

    Driven directly rather than through a campaign: the campaign proves today's
    writers comply, and this proves tomorrow's non-compliant one is caught.
    """
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "summary.md").write_text("# Summary\n\nfrom the DB\n")

    assert unexplained_views(tmp_path) == ["/nested/summary.md"]


def test_a_stamped_view_and_a_rebuilt_one_are_both_accepted(tmp_path):
    """Both of criterion 4's options, so the check cannot pass by rejecting one."""
    stamped = derived_note(source_of_record="x.json", warning="read the json")
    (tmp_path / "stamped.md").write_text(stamped + "\n\n# Title\n")
    overlay = tmp_path / ".labpilot" / "skills"
    overlay.mkdir(parents=True)
    (overlay / "agent.md").write_text("- Keep: nothing\n")

    assert unexplained_views(tmp_path) == []


def test_every_rebuilt_directory_names_something_that_exists():
    """A rebuilder that has been renamed leaves its directory permanently
    exempt, which is how an exemption outlives the thing that justified it."""
    from labpilot.research_engine.evidence import overlay_repair

    for fragment, rebuilder in REDERIVED.items():
        assert hasattr(overlay_repair, rebuilder), f"{fragment} names a missing {rebuilder}"
