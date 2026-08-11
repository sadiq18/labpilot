"""One way to say *"this file is a copy, and here is what it is a copy of"*.

M20 criterion 4. Three defects were one shape: a file written once from a source
that later changed, then read as if it were the source. Plan projections said
`ready` for all 19 plans while the DB said `done=16, abandoned=3`; overlays told
six agents to avoid the only technique that ever improved the metric, because the
card behind that lesson had been repaired and the overlay had no path back to it.
Two wrong diagnoses came from reading those files, six days apart, by the same
author — and both were reasonable, because every file agreed with every other.

A derived artifact has two honest options: re-derive when it is read, or say what
it is. This is the second one, and it is one helper rather than a copy per writer
because two implementations of a single idea drifting apart is the defect
criterion 2 is named after — `projection_stamp` was already the first copy.

The stamp carries three facts, and the middle one is the one that does the work:
that this is not authoritative, **what to read instead**, and when it was taken.
A warning without a source of record tells a reader to distrust the file and not
where to go, which is the position the two misdiagnoses were already in.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

#: How a stamped markdown view begins, so a machine reader can drop it again.
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
    """The same facts for a view that is text rather than JSON.

    A blockquote, so it renders as a callout rather than as body prose — the
    reader who was misled by the plan projections was reading markdown, and a
    provenance line indistinguishable from the content is one they scroll past.

    The generic text says what is true of every view — that it is a copy, and of
    what — and leaves *why it might be wrong* to the caller's `warning`. It said
    "and may have changed since" for one round, which is false for a view written
    in the same call as its source: `comparison.md` and `profile.md` are always
    exactly as fresh as the JSON beside them. A stamp that overstates is the same
    defect as one that misdirects, and this file's whole subject is a document
    asserting something it cannot know. Reported reviewing this branch.

    `dated` is opt-in because a timestamp makes the renderer non-deterministic,
    and some of these are documented as pure functions of their input: the
    comparator says *"Deterministic markdown view"* and a test asserts two renders
    of one comparison are byte-identical. Stamping unconditionally broke it — and
    the property is worth keeping, since a view regenerated identically produces
    no diff to review.

    Take the date where staleness *over time* is the danger, as it is for plan
    projections: the DB moves under them and nothing rewrites the file. Leave it
    where the danger is the source being rewritten in place, as it is for a
    comparison — there "read the JSON" is the fact that acts, and the JSON's own
    mtime already says when.
    """
    when = datetime.now(UTC).isoformat() if dated else ""
    dateline = f"Generated {when} from" if dated else "Generated from"
    return (
        f"> **Derived view — not authoritative.** {dateline} "
        f"`{source_of_record}`, which is the source of record.\n>\n> {warning}"
    )


def strip_derived_note(text: str) -> str:
    """The content without its provenance block.

    A stamp is for a human deciding whether to trust the file. Two readers of
    `research_brief.md` are not human — `planner.py` feeds it to an LLM under a
    2000-character budget — and for them the block is 200 characters of overhead
    that displaces the brief it is attached to. Stripping costs nothing and keeps
    the stamp free.

    Only a *leading* block, and only the blockquote plus the blank line after it,
    so a view whose own content contains a quote keeps it.
    """
    if not text.lstrip().startswith(_NOTE_OPENER):
        return text
    lines = text.lstrip().splitlines()
    index = 0
    while index < len(lines) and lines[index].startswith(">"):
        index += 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    return "\n".join(lines[index:])


def read_derived(path: Path | str, *, errors: str = "strict") -> str:
    """A persisted view's content, without the block a machine does not need.

    Every reader of `research_brief.md` that feeds an LLM wants this, and each
    one had to know to ask: of its four readers two stripped and two did not —
    including the codegen prompt, the one role the comments call *"must never
    degrade"*, which spent ~250 of its 3000 characters telling the model to
    distrust the context it was being handed. Reported reviewing this branch.

    A shared reader rather than a shared stripper, because the next consumer will
    write `path.read_text()` and not think about provenance at all.
    """
    return strip_derived_note(Path(path).read_text(encoding="utf-8", errors=errors))
