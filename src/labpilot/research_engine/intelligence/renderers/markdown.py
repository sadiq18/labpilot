"""Markdown renderer for the durable Research Brief."""

from __future__ import annotations

from pathlib import Path

from labpilot.research_engine.intelligence.brief.models import ResearchBrief

_SECTIONS: tuple[tuple[str, str], ...] = (
    ("Problem summary", "problem_summary"),
    ("Dataset overview", "dataset_overview"),
    ("Competition rules & metric", "rules_and_metric"),
    ("Related papers", "related_papers"),
    ("Similar competitions", "similar_competitions"),
    ("Relevant GitHub repositories", "repositories"),
    ("Winning techniques", "winning_techniques"),
    ("Existing beliefs", "beliefs"),
    ("Top hypotheses", "top_hypotheses"),
    ("Known risks", "known_risks"),
    ("Suggested next experiments", "suggested_experiments"),
)


def render_brief_markdown(brief: ResearchBrief) -> str:
    """Render headed markdown sections in briefing order."""
    lines = ["# Research Brief", ""]
    payload = brief.model_dump(mode="json")
    for heading, key in _SECTIONS:
        lines.append(f"## {heading}")
        lines.append("")
        value = payload.get(key)
        if isinstance(value, list):
            if value:
                lines.extend(f"- {item}" for item in value)
            else:
                lines.append("_(none)_")
        else:
            text = str(value or "").strip()
            lines.append(text if text else "_(unavailable)_")
        lines.append("")
    lines.append("---")
    lines.append(f"_generated_by={brief.generated_by}_")
    if brief.notes:
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        lines.extend(f"- {note}" for note in brief.notes)
        lines.append("")
    return "\n".join(lines)


def write_brief(brief: ResearchBrief, path: Path) -> Path:
    """Write ``research_brief.md`` (creating parent dirs) and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_brief_markdown(brief) + "\n")
    return path
