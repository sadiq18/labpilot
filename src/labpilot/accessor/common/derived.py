"""One way to say *"this file is a copy, and here is what it is a copy of"*.

M20 criterion 4: a derived artifact either re-derives when it is read, or says
what it is. This is the second option, in one place rather than a copy per
writer. The reasoning, the four defects that produced the rule, and the reviews
that shaped it are in `docs/research-os/autonomy-roadmap/15-gates-must-fail.md`.

A stamp carries three facts, and the middle one does the work: that this is not
authoritative, **what to read instead**, and when it was taken. A warning with no
source of record tells a reader to distrust the file without saying where to go.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

#: How a stamped markdown view begins. `derived_note` writes it and
#: `strip_derived_note` matches it, so it exists once rather than as two literals
#: that have to stay byte-identical.
_NOTE_OPENER = "> **Derived view — not authoritative.**"


def derived_stamp(*, source_of_record: str, warning: str) -> dict[str, object]:
    """Provenance block for a JSON view: not authoritative, and what is."""
    return {
        "authoritative": False,
        "source_of_record": source_of_record,
        "generated_at": datetime.now(UTC).isoformat(),
        "warning": warning,
    }


def derived_note(*, source_of_record: str, warning: str, dated: bool = False) -> str:
    """The same facts as `derived_stamp`, for a view that is text not JSON.

    Rendered *from* the stamp so the fact set has one definition: a field added
    there reaches this too, instead of reaching the JSON views only.

    A blockquote, so it reads as a callout rather than as body prose. `dated` is
    opt-in — take the date where staleness over time is the danger, as it is for
    plan projections; leave it where the source is rewritten in place and the
    warning is what acts.
    """
    stamp = derived_stamp(source_of_record=source_of_record, warning=warning)
    dateline = f"Generated {stamp['generated_at']} from" if dated else "Generated from"
    return (
        f"{_NOTE_OPENER} {dateline} `{stamp['source_of_record']}`, "
        f"which is the source of record.\n>\n> {stamp['warning']}"
    )


def strip_derived_note(text: str) -> str:
    """The content without its provenance block.

    A stamp is for a human deciding whether to trust the file; the readers that
    feed `research_brief.md` to a model are not, and for them the block is
    ~250 characters displacing the brief inside a 2000-character budget.

    Only a *leading* block, and only the blockquote plus the blank line after it,
    so a view whose own content contains a quote keeps it.
    """
    # Stamped files start with the opener, so the common paths — a stamped view,
    # or JSON that cannot be one — do no work beyond a prefix test.
    body = text if text.startswith(_NOTE_OPENER) else text.lstrip()
    if not body.startswith(_NOTE_OPENER):
        return text

    lines = body.splitlines()
    index = 0
    while index < len(lines) and lines[index].startswith(">"):
        index += 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    return "\n".join(lines[index:])


def read_derived(path: Path | str, *, errors: str = "strict") -> str:
    """A persisted view's content, without the block a machine does not need.

    A shared reader rather than a shared stripper: the next consumer will write
    `path.read_text()` and not think about provenance at all.
    """
    return strip_derived_note(Path(path).read_text(encoding="utf-8", errors=errors))
