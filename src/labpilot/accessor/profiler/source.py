"""Where a dataset lives, separated from what is inferred about it.

M22 step 1. Until now the profiler reached for the filesystem itself —
``data_dir.rglob("*.csv")``, ``pd.read_csv(path, nrows=...)`` — nine times
across two code paths. That is why every inference it makes is quietly a
statement about *files*, and why a warehouse table, an object store or an
interactive environment could not be described by it at all
([M12](../../../docs/research-os/autonomy-roadmap/06-beyond-kaggle.md)).

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
from pydantic import BaseModel

__all__ = [
    "DatasetSource",
    "DeclaredFacts",
    "LocalFileSource",
    "TableRef",
]


class TableRef(BaseModel):
    """One addressable table, wherever it lives."""

    #: Stable within a source. For files this is the path relative to the
    #: dataset root, which is also what the profile records.
    id: str
    #: How the source reaches it: a relative path here, a ``sql://`` or
    #: ``s3://`` locator in an adapter that has one.
    uri: str


class DeclaredFacts(BaseModel):
    """What the environment states about itself before anything is inferred.

    A source of *signals*, never of values. A declaration the data contradicts
    is evidence that something is wrong, not an instruction — which is why this
    is returned by the source rather than written onto the schema.

    Only what a consumer reads today lives here. The target, metric and id
    declarations that step 3 will weigh arrive with the code that weighs them.
    """

    title: str = ""
    description: str = ""


@runtime_checkable
class DatasetSource(Protocol):
    """Read access to a dataset's tables. No inference, no side effects."""

    def tables(self) -> list[TableRef]: ...

    def columns(self, table: TableRef) -> list[str]:
        """Column names only — the cheapest question, asked constantly."""
        ...

    def sample(self, table: TableRef, limit: int | None = None) -> pd.DataFrame:
        """Up to ``limit`` units, or all of them when ``limit`` is None."""
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
            TableRef(id=str(path.relative_to(self.root)), uri=str(path.relative_to(self.root)))
            for path in sorted(self.root.rglob("*.csv"))
        ]

    def path(self, table: TableRef) -> Path:
        """The absolute path behind a table — the one place a uri becomes one.

        Local-file only, and deliberately not part of :class:`DatasetSource`: a
        warehouse table has no path. Every read below goes through it, and it is
        public so that a caller which genuinely needs a filesystem path asks
        here rather than re-deriving `root / uri` for itself.
        """
        return self.root / table.uri

    def columns(self, table: TableRef) -> list[str]:
        return [str(column) for column in pd.read_csv(self.path(table), nrows=0).columns]

    def sample(self, table: TableRef, limit: int | None = None) -> pd.DataFrame:
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
