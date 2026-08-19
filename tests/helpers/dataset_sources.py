"""Dataset sources that are not directories.

M22. `LocalFileSource` is the only adapter that ships, so the seam is only
as real as the tests that drive something else through it. `DictSource` is
that something else: frames in memory, no root, no paths — so any read that
reaches for the filesystem fails loudly instead of quietly working because
the test happened to run beside real files.
"""

from __future__ import annotations

import pandas as pd

from labpilot.accessor.profiler.source import DeclaredFacts, TableRef

__all__ = ["DictSource"]


class DictSource:
    """A dataset that is not a directory. Implements `DatasetSource`, nothing more.

    Deliberately holds no path, so any filesystem read the profiler attempts
    fails loudly rather than silently working because the test happened to run
    beside real files.
    """

    def __init__(self, frames: dict[str, pd.DataFrame], declared: DeclaredFacts | None = None):
        self._frames = frames
        self._declared = declared or DeclaredFacts()

    def tables(self) -> list[TableRef]:
        return [TableRef(uri=name) for name in self._frames]

    def columns(self, table: TableRef) -> list[str]:
        return [str(column) for column in self._frames[table.uri].columns]

    def sample(self, table: TableRef, limit: int | None) -> pd.DataFrame:
        frame = self._frames[table.uri]
        return frame if limit is None else frame.head(limit)

    def exact_unit_count(self, table: TableRef, column: str) -> int:
        return len(self._frames[table.uri])

    def declared(self) -> DeclaredFacts:
        return self._declared
