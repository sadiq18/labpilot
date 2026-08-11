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

#: Key under which a JSON view carries its stamp. Plan projections predate this
#: and keep `_projection`, which is already on disk in every workspace.
DERIVED_KEY = "_derived"


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
        f"`{source_of_record}`, which is the source of record and may have "
        f"changed since.\n>\n> {warning}"
    )
