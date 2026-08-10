"""Write-then-rename so a reader never observes a partial file (M11).

`Path.write_text()` opens with truncate, then writes — two separate steps, not
one. A reader landing between them sees a 0-byte (or partial) file. Renaming a
finished temp file onto the real path is atomic on POSIX (`os.replace`):
any reader sees either the whole old file or the whole new one, never
something in between, with no lock required on the read side.

A process killed between the temp write and the rename leaves that temp file
behind permanently — no reconciliation sweep exists to remove it. Harmless
today (the `.tmp-<pid>-<tid>` name never matches any store's real-file glob,
so it's inert clutter, not a correctness bug), just unswept. Worth a cleanup
pass if this ever accumulates enough to matter in practice; not built
speculatively here.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path


def atomic_write_text(path: str | Path, content: str, *, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    tmp_path.write_text(content, encoding=encoding)
    tmp_path.replace(path)  # atomic on POSIX — same filesystem, same directory
