"""What a captured competition is, and what it admits it cannot prove.

M24. A fixture stands in for a real dataset, and the whole question is *when it
is allowed to*. Two rules make that answerable rather than hopeful:

* **Every file records where it came from** — ``source_sha256``, ``source_bytes``,
  ``source_rows``, ``fixture_rows`` — so a fixture can be re-derived from a
  re-download and proven identical. `tests/fixtures/real_failures/MANIFEST.md`
  exists because a 79-byte paraphrase once sat in a corpus claiming to be 624
  bytes; the same discipline, one layer up.
* **Every fixture declares what truncation destroyed.** `unverifiable` names a
  criterion and the reason, and the scorer *refuses to score* those rather than
  scoring a truncation artifact. A capture that kept no rows cannot speak to
  cardinality, and a scorer that reported `fail` there would be measuring the
  capture, not the profiler.

Headers alone buy target, id, submission columns, train-only columns and file
roles — five criteria, at ~400 bytes and no licence risk. Rows buy four more
things: dtype, cardinality, anchor equality, suffix contiguity. So rows are a
per-fixture justification, not a policy.

Plan: ``docs/research-os/autonomy-roadmap/19-competition-benchmark.md``
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "CAPTURE_MODES",
    "STALE_AFTER_DAYS",
    "CapturedFile",
    "CompetitionFixture",
    "Expectations",
    "KnownFailure",
    "load_fixture",
    "save_fixture",
]

#: How much of a file a fixture keeps.
#:
#: * ``verbatim`` — the whole file. Only for files small enough to be one.
#: * ``headers_only`` — the header line. The default, and enough for five
#:   criteria.
#: * ``head:<n>`` — the first n data rows. Cheap, and it destroys any invariant
#:   that depends on absolute row position.
#: * ``stride:<k>`` — every kth row, ids renumbered. Preserves contiguous
#:   prefixes and suffixes, equality-where-known, per-kind column sets and
#:   dtypes; changes counts and numeric statistics, which are declared
#:   unverifiable anyway. rogii needs this: its scored suffixes start at rows
#:   1442/1545/2083 and `_detect_suffix_scoring` reads absolute indices, so
#:   `head:50` would not truncate the fact — it would invert it.
CAPTURE_MODES = ("verbatim", "headers_only", "head", "stride")


class CapturedFile(BaseModel):
    """One file of the real dataset, and what was kept of it."""

    path: str
    mode: str
    #: Of the **source** file, always — so a re-download can be checked against
    #: it even when the fixture holds three of its rows.
    source_sha256: str
    source_bytes: int
    source_rows: int | None = None
    fixture_rows: int | None = None


#: How long a declared defect may sit unreviewed before the corpus calls it stale.
#:
#: Not a deadline for the *fix* — plenty of defects are worth carrying for a
#: year. A deadline for the *claim*: the reason names a cause, and a cause
#: nobody has re-read in six months describes code that may no longer exist.
STALE_AFTER_DAYS = 180


class KnownFailure(BaseModel):
    """A criterion this fixture ships red on purpose, and since when.

    Both fields are load-bearing. The reason is what makes a red cell a decision
    rather than a defect nobody noticed. The date is what makes that decision
    reviewable — without one, *"we know about it"* is unfalsifiable, and an
    xfail that has outlived its cause reads exactly like one that has not.

    M24 exit criterion 5: *"a stale xfail is a lie about intent."*
    """

    reason: str = Field(min_length=1)
    #: When the failure was **declared**, not when the defect was introduced.
    #: The reason may name an older date — s6e7's stale spec was written three
    #: weeks before anyone recorded that it was wrong — and conflating the two
    #: would age a claim by the lifetime of the bug it describes.
    declared: date

    @model_validator(mode="before")
    @classmethod
    def _refuse_a_bare_reason(cls, value: object) -> object:
        """The pre-M24 shape was `criterion → reason`. Say what is missing.

        Accepting a bare string here would be the easy migration and would also
        defeat the criterion: every fixture written afterwards would carry a
        reason and no date, which is the state this model exists to end.
        """
        if isinstance(value, str):
            raise ValueError(
                "a known failure needs a date as well as a reason — "
                '{"reason": ..., "declared": "YYYY-MM-DD"}. '
                "A reason nobody can date cannot be reviewed."
            )
        return value

    def age_days(self, today: date) -> int:
        """Days since the claim was made. `today` is passed in, never read."""
        return (today - self.declared).days

    def is_stale(self, today: date) -> bool:
        return self.age_days(today) > STALE_AFTER_DAYS


class Expectations(BaseModel):
    """What the right answer is, per criterion.

    `None` means *not applicable to this competition* — a dataset with no
    scoring input has no `train_test_relationship` to get right — and is scored
    as such rather than as a miss.
    """

    target_column: str | None = None
    id_columns: list[str] | None = None
    train_test_relationship: str | None = None
    modality: str | None = None
    feature_columns: list[str] | None = None
    metric_name: str | None = None
    #: Fields the system is expected to **refuse to answer**. A fixture whose
    #: right outcome is "should have asked" is as valuable as one with a known
    #: answer, and is scored on that alone — otherwise adding a genuinely hard
    #: competition tanks the accuracy number, which creates pressure to guess.
    must_ask: list[str] = Field(default_factory=list)


class CompetitionFixture(BaseModel):
    """A captured competition, and the terms on which it may be scored."""

    slug: str
    captured_at: str
    #: Where it came from, for a human. Never read by the harness.
    source: str = ""
    #: ``verbatim`` when every file is; ``derived`` the moment one is not.
    provenance: Literal["verbatim", "derived"] = "derived"
    #: Where `listing.tsv` came from, and therefore what it proves.
    #:
    #: ``walked`` means a local directory was read and every listed file carries
    #: a sha256 of bytes this tool saw. ``remote`` means the Kaggle API reported
    #: names and sizes and nothing else — the counts and ratios modality
    #: detection needs, and no evidence about content.
    #:
    #: Recorded rather than inferred from an empty sha column, because "no
    #: checksum" and "checksum of an empty file" are one character apart in a
    #: TSV and worlds apart as provenance.
    listing_source: Literal["walked", "remote", "none"] = "none"
    licence: str = "unknown"
    #: Whether the source dataset's **rows** may be redistributed.
    #:
    #: ``forbidden`` is not a statement about this fixture — it is a constraint
    #: *on* it, and an enforced one: a forbidden fixture may carry column names
    #: and no data rows. That is what makes the corpus usable for a sponsored or
    #: private dataset, and what stopped this field from being a note that
    #: contradicted the commit containing it.
    redistribution: Literal["allowed", "forbidden", "unknown"] = "unknown"
    files: list[CapturedFile] = Field(default_factory=list)
    expected: Expectations = Field(default_factory=Expectations)
    #: criterion → why the capture cannot speak to it. The scorer reports
    #: `unverifiable`, which is not a pass and not a failure.
    unverifiable: dict[str, str] = Field(default_factory=dict)
    #: criterion → what is wrong today, and since when. A fixture that ships red
    #: on purpose, so the day it goes green is visible instead of silent.
    known_failures: dict[str, KnownFailure] = Field(default_factory=dict)
    notes: str = ""

    @property
    def carries_rows(self) -> bool:
        """Whether any captured file holds a data row."""
        return any((entry.fixture_rows or 0) > 0 for entry in self.files)

    @property
    def honours_its_licence(self) -> bool:
        """A fixture forbidden from redistributing rows carries none."""
        return self.redistribution != "forbidden" or not self.carries_rows


FIXTURE_FILENAME = "fixture.json"


def load_fixture(directory: Path) -> CompetitionFixture:
    path = Path(directory) / FIXTURE_FILENAME
    return CompetitionFixture.model_validate(json.loads(path.read_text(encoding="utf-8")))


def save_fixture(directory: Path, fixture: CompetitionFixture) -> Path:
    path = Path(directory) / FIXTURE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fixture.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path
