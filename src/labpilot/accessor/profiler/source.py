"""Where a dataset lives, separated from what is inferred about it.

M22 step 1. Until now the profiler reached for the filesystem itself —
``data_dir.rglob("*.csv")``, ``pd.read_csv(path, nrows=...)`` — nine times
across two code paths. That is why every inference it makes is quietly a
statement about *files*, and why a warehouse table, an object store or an
interactive environment could not be described by it at all
([M12](../../../../docs/research-os/autonomy-roadmap/06-beyond-kaggle.md)).

The split this module introduces:

* a **source** answers *what tables exist and how do I read them* — and nothing
  else. It has no opinion about targets, ids, splits or metrics;
* the **profiler** infers, and reads only through a source.

``LocalFileSource`` is the only implementation, deliberately. A seam with one
implementation is honest; three unused adapters would be the "seventh provider
adapter" the roadmap warns about. What the seam buys today is that every read
has one owner; what it buys later is that M12 writes an adapter instead of a
profiler.

**Not here yet, by design.** ``TableRef`` carries no ``role``: deciding that a
table is train, test or a prediction template is *inference* — it is how
``train_test_relationship`` is answered — and it lands in step 3 with the
evidence that justifies it. A field the profiler cannot yet fill would be a
declaration nothing reaches, which is the defect class this milestone removes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd
from pydantic import BaseModel, Field

from labpilot.accessor.profiler.schema import MetricRef

__all__ = [
    "DatasetSource",
    "DeclaredFacts",
    "LocalFileSource",
    "TableRef",
]


class TableRef(BaseModel):
    """One addressable table, wherever it lives.

    One field, because a `id` alongside it was the same string at every
    construction site — two spellings of one fact, which is the pattern this
    milestone's own design forbids. An adapter that needs an identity distinct
    from its locator adds the field then, with something to distinguish.
    """

    #: How the source reaches it: a path relative to the dataset root here, a
    #: ``sql://`` or ``s3://`` locator in an adapter that has one. For files
    #: this is also what the profile records.
    uri: str


class DeclaredFacts(BaseModel):
    """What the environment states about itself before anything is inferred.

    A source of *signals*, never of values. A declaration the data contradicts
    is evidence that something is wrong, not an instruction — which is why this
    is returned by the source rather than written onto the schema.

    Only what a consumer reads today lives here.
    """

    title: str = ""
    description: str = ""
    #: Field name to value, from `schema_answers.json` — what a human has
    #: already settled about this dataset. Carried by the source rather than
    #: read by the profiler because the profiler has no workspace: a warehouse
    #: adapter answers the same questions from wherever it keeps them.
    answers: dict[str, str] = Field(default_factory=dict)
    #: What the environment says it scores by. Already canonical: the caller
    #: resolves the name through the metric registry, because `accessor` may not
    #: import `research_engine`.
    metric: MetricRef | None = None


@runtime_checkable
class DatasetSource(Protocol):
    """Read access to a dataset's tables. No inference, no side effects."""

    def tables(self) -> list[TableRef]: ...

    def columns(self, table: TableRef) -> list[str]:
        """Column names only — the cheapest question, asked constantly."""
        ...

    def sample(self, table: TableRef, limit: int | None) -> pd.DataFrame:
        """Up to ``limit`` units, or all of them when ``limit`` is None.

        `limit` has no default. Reading a whole table is a decision — 690,088
        rows for the one competition that has already cost this repo a campaign
        — and a default makes it one a caller can take by omission.
        """
        ...

    def exact_unit_count(self, table: TableRef, column: str) -> int:
        """The true count, paid for by reading one column rather than all."""
        ...

    def declared(self) -> DeclaredFacts: ...


class LocalFileSource:
    """A directory of CSVs.

    Table order is ``sorted(root.rglob("*.csv"))`` — the order the profile's
    ``files`` list has always had, kept because a profile that reorders its own
    description between runs cannot be compared against itself.
    """

    #: Rows per chunk when counting. Large enough that the per-chunk overhead
    #: disappears, small enough that one chunk of one column is not a memory
    #: event on a wide table.
    _COUNT_CHUNK = 10_000

    def __init__(self, root: Path, declared: DeclaredFacts | None = None) -> None:
        self.root = Path(root)
        self._declared = declared or DeclaredFacts()

    def tables(self) -> list[TableRef]:
        return [
            TableRef(uri=str(path.relative_to(self.root)))
            for path in sorted(self.root.rglob("*.csv"))
        ]

    def path(self, table: TableRef) -> Path:
        """The absolute path behind a table — the one place a uri becomes one.

        Local-file only, and deliberately not part of :class:`DatasetSource`: a
        warehouse table has no path. Every read below goes through it, and it is
        public so that a caller which genuinely needs a filesystem path asks
        here rather than re-deriving `root / uri` for itself.

        **Refuses to leave the root.** `Path.__truediv__` discards its left
        operand when the right one is absolute, so ``root / "/etc/passwd"`` is
        ``/etc/passwd`` — no error, no trace. Every uri today comes from
        :meth:`tables`, which cannot produce one; from step 3 they will also
        come from operator answers and model proposals, and the check has to
        exist before the untrusted caller does, not after.
        """
        root = self.root.resolve()
        resolved = (root / table.uri).resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(
                f"table uri {table.uri!r} resolves outside the dataset root {self.root}"
            )
        return resolved

    def columns(self, table: TableRef) -> list[str]:
        return [str(column) for column in pd.read_csv(self.path(table), nrows=0).columns]

    def sample(self, table: TableRef, limit: int | None) -> pd.DataFrame:
        return pd.read_csv(self.path(table), nrows=limit)

    def exact_unit_count(self, table: TableRef, column: str) -> int:
        return sum(
            len(chunk)
            for chunk in pd.read_csv(
                self.path(table), usecols=[column], chunksize=self._COUNT_CHUNK
            )
        )

    def physical_line_count(self, table: TableRef) -> int:
        """Lines in the file, header included.

        Not :meth:`exact_unit_count`, and deliberately not part of the protocol:
        it counts physical lines, so a quoted newline inside a field makes it
        wrong. Suffix-scoring detection has always counted this way and its
        result feeds `scored_fraction`; changing the measurement belongs to the
        step that makes row counts honest, not to the step that moves reads.
        """
        with self.path(table).open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)

    def declared(self) -> DeclaredFacts:
        return self._declared
