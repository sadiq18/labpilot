"""Cross-process exclusive lock over a read-then-write on disk (M11).

Not process-local like a `threading.Lock` — branches under M11's parallel
fan-out are separate git worktrees and may end up as separate OS processes,
not just threads, so the lock has to work across process boundaries too.
`fcntl.flock` does; a `threading`/`multiprocessing` lock would silently stop
protecting anything the moment execution moves off threads.

Deliberately does not delete the lock file after use: unlinking a path while
another caller may already have it open is a TOCTOU hazard of its own — the
unlinker's `flock` releases on close, but a racing caller that opened the
*old* (now-unlinked) inode before the unlink still holds a "lock" that a
fresh caller opening a *new* inode at the same path can't see, defeating
mutual exclusion for the two of them. The lock file is a few bytes and one
is created per distinct key ever locked (e.g. per hypothesis id) — bounded by
that key space, not by call volume, and not worth the correctness risk to
reclaim.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def locked(lock_path: str | Path) -> Iterator[None]:
    """Hold an exclusive lock at `lock_path` for the duration of the block.

    `fcntl` is imported lazily (not at module import time) so that importing
    a module which merely defines locked sections — but doesn't execute one —
    still works on a non-POSIX platform; only actually entering this context
    manager requires `fcntl`.
    """
    import fcntl

    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
