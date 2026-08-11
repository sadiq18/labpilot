"""Packaged skill.md + competition-local skill overlays for micro agents."""

from __future__ import annotations

import re
from pathlib import Path

from labpilot.accessor.common.derived import derived_note, strip_derived_note

DEFAULT_OVERLAY_DIRNAME = ".labpilot/skills"
#: Soft budget before rewriting the on-disk overlay to a compact summary.
ON_DISK_CHAR_BUDGET = 5000
#: Hard cap when injecting into a model prompt.
INJECT_CHAR_BUDGET = 1800


def packaged_skill_path(agent_file: str | Path) -> Path | None:
    """Return ``skill.md`` beside the agent's module file, if present."""
    path = Path(agent_file).resolve().parent / "skill.md"
    return path if path.is_file() else None


def load_packaged_skill(agent_file: str | Path, *, max_chars: int = 6000) -> str:
    path = packaged_skill_path(agent_file)
    if path is None:
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return _clip(text, max_chars)


def overlay_dir(workspace_root: Path | str | None) -> Path | None:
    if not workspace_root:
        return None
    return Path(workspace_root) / DEFAULT_OVERLAY_DIRNAME


def overlay_path(workspace_root: Path | str | None, agent_key: str) -> Path | None:
    root = overlay_dir(workspace_root)
    if root is None:
        return None
    safe = re.sub(r"[^a-zA-Z0-9_\-]+", "_", agent_key.strip().lower()) or "agent"
    return root / f"{safe}.md"


#: Overlays are rebuilt from the cards only in part — `Try:` and `Note:` lines
#: are prose the repair pass deliberately preserves — so they carry a stamp.
OVERLAY_NOTE = derived_note(
    source_of_record="research/evidence/ (the evidence cards)",
    warning="Lessons here are rewritten when their card is repaired; the notes are not.",
)


def overlay_body(text: str) -> str:
    """Overlay content without its provenance note."""
    return strip_derived_note(text)


def stamped_overlay(body: str) -> str:
    """Overlay content with its provenance note."""
    return OVERLAY_NOTE + "\n\n" + body.lstrip("\n")


def load_skill_overlay(
    workspace_root: Path | str | None,
    agent_key: str,
    *,
    max_chars: int = INJECT_CHAR_BUDGET,
) -> str:
    """Bounded overlay text for prompt injection (never dumps huge files)."""
    path = overlay_path(workspace_root, agent_key)
    if path is None or not path.is_file():
        return ""
    text = overlay_body(path.read_text(encoding="utf-8", errors="replace")).strip()
    if len(text) <= max_chars:
        return text
    return summarize_skill_text(text, max_chars=max_chars)


def upsert_skill_overlay(
    workspace_root: Path | str,
    agent_key: str,
    *,
    lesson_id: str,
    keep: list[str] | None = None,
    avoid: list[str] | None = None,
    try_next: list[str] | None = None,
    note: str = "",
    on_disk_budget: int = ON_DISK_CHAR_BUDGET,
) -> Path:
    """Idempotently append a keyed lesson; summarize on disk when too long."""
    path = overlay_path(workspace_root, agent_key)
    assert path is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = (
        overlay_body(path.read_text(encoding="utf-8", errors="replace")) if path.is_file() else ""
    )
    marker = f"<!-- lesson:{lesson_id} -->"
    if marker in existing:
        return path

    block_lines = [marker, f"## Lesson `{lesson_id}`"]
    for label, items in (
        ("Keep", keep or []),
        ("Avoid", avoid or []),
        ("Try", try_next or []),
    ):
        for item in items:
            text = str(item).strip()
            if text:
                block_lines.append(f"- {label}: {text}")
    if note.strip():
        block_lines.append(f"- Note: {note.strip()[:400]}")
    block_lines.append("")
    updated = (existing.rstrip() + "\n\n" if existing.strip() else "") + "\n".join(
        block_lines
    )
    if len(updated) > on_disk_budget:
        updated = summarize_skill_text(updated, max_chars=on_disk_budget)
    # Stamped after summarising, so the note is never what gets truncated.
    path.write_text(stamped_overlay(updated.rstrip() + "\n"), encoding="utf-8")
    return path


def summarize_skill_text(text: str, *, max_chars: int) -> str:
    """Compress overlay / skill prose into keep/avoid/try bullets + recent lessons."""
    keep = _collect_bullets(text, "Keep")
    avoid = _collect_bullets(text, "Avoid")
    try_next = _collect_bullets(text, "Try")
    lesson_ids = re.findall(r"<!-- lesson:([^ ]+) -->", text)
    parts = ["# Competition skill overlay (summarized)", ""]
    if keep:
        parts.append("## Keep")
        parts.extend(f"- {item}" for item in keep[:12])
        parts.append("")
    if avoid:
        parts.append("## Avoid")
        parts.extend(f"- {item}" for item in avoid[:12])
        parts.append("")
    if try_next:
        parts.append("## Try")
        parts.extend(f"- {item}" for item in try_next[:12])
        parts.append("")
    if lesson_ids:
        parts.append("## Recent lesson ids")
        parts.append("- " + ", ".join(lesson_ids[-8:]))
        parts.append("")
    summary = "\n".join(parts).strip()
    if len(summary) <= max_chars:
        return summary
    return _clip(summary, max_chars)


def compose_system_prompt(
    base_prompt: str,
    *,
    agent_file: str | Path | None = None,
    workspace_root: Path | str | None = None,
    agent_key: str = "",
    inject_budget: int = INJECT_CHAR_BUDGET,
) -> str:
    """Merge packaged skill.md + bounded competition overlay into system prompt."""
    chunks = [base_prompt.strip()]
    if agent_file is not None:
        packaged = load_packaged_skill(agent_file)
        if packaged:
            chunks.append("# Packaged skill\n" + packaged)
    key = agent_key or (
        Path(agent_file).parent.name if agent_file is not None else ""
    )
    overlay = load_skill_overlay(workspace_root, key, max_chars=inject_budget)
    if overlay:
        chunks.append("# Competition skill overlay\n" + overlay)
    return "\n\n".join(c for c in chunks if c).strip()


def _collect_bullets(text: str, label: str) -> list[str]:
    pattern = re.compile(rf"^[-*]\s*{label}:\s*(.+)$", re.I | re.M)
    seen: list[str] = []
    for match in pattern.finditer(text):
        item = match.group(1).strip()
        if item and item not in seen:
            seen.append(item)
    return seen


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
