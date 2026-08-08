"""Per-execution code provenance — M19 §6.

A delta is ``parent + change``, so it needs an addressable parent. The
experiment graph *branches* — H-1 and H-2 both fork from the baseline — while
the filesystem does not: there is one ``pipeline/train.py``, overwritten in
place.

The existing backup at ``artifacts/code_backups/train_<execution_id>.py`` looks
like it solves this and does not. It stores the file that existed *before* that
execution wrote its own, so it is keyed by the **consumer**, not the producer:
it answers "what did E-007 start from", never "what did E-007 run". Recovering
the latter means finding whichever execution wrote next and reading *its*
backup, which is inference from ordering — and ordering is exactly what a
branching graph does not give you. Two children forking from one parent break
it outright.

So each execution's *resulting* source is snapshotted under
``runs/<execution_id>/``, keyed by the execution that produced it. The workspace
tree stays the working copy; this is the addressable record. Then "the code my
parent ran" is a lookup rather than a reconstruction.

Snapshots are written after ``apply_proposal`` succeeds, so what is recorded is
what was actually applied — not what was proposed. A rejected proposal leaves no
snapshot, which is correct: nothing ran.

Failure here is logged and swallowed. This is provenance, and losing it must not
cost a training run that is otherwise fine.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

#: Workspace-relative root holding one directory per execution.
SNAPSHOT_ROOT = "runs"

#: The file every consistency check cares about, relative to a snapshot.
TRAIN_RELPATH = "pipeline/train.py"


def snapshot_dir(workspace_root: Path | str, execution_id: str) -> Path:
    """Directory holding the source ``execution_id`` ran."""
    return Path(workspace_root) / SNAPSHOT_ROOT / str(execution_id)


def record_execution_source(
    workspace_root: Path | str,
    execution_id: str,
    written: list[Path],
) -> list[Path]:
    """Snapshot ``written`` under ``runs/<execution_id>/``, mirroring layout.

    Returns the snapshot paths actually created. Relative layout is preserved,
    so a file applied to ``pipeline/train.py`` lands at
    ``runs/<execution_id>/pipeline/train.py`` and reads back by the same key.

    Re-running an execution overwrites its snapshot: the record should reflect
    the code that ran, and on a retry that is the last applied attempt.
    """
    root = Path(workspace_root)
    if not execution_id:
        # Without a key there is nothing to address the snapshot by, and a
        # snapshot nobody can look up is worse than none — it takes disk and
        # answers no question.
        logger.warning("Skipping code snapshot: no execution id")
        return []

    target_root = snapshot_dir(root, execution_id)
    created: list[Path] = []
    for path in written:
        try:
            rel = Path(path).resolve().relative_to(root.resolve())
        except ValueError:
            # Written outside the workspace: not ours to record.
            logger.warning("Skipping code snapshot for %s: outside workspace", path)
            continue
        target = target_root / rel
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(path).read_bytes())
        except OSError as exc:
            logger.warning("Could not snapshot %s for %s: %s", rel, execution_id, exc)
            continue
        created.append(target)
    return created


def execution_source(
    workspace_root: Path | str,
    execution_id: str,
    relpath: str = TRAIN_RELPATH,
) -> str | None:
    """Source ``execution_id`` ran for ``relpath``, or None when unrecorded.

    None is the honest answer for an execution that predates snapshotting or
    whose proposal was rejected. Callers must treat it as "unknown", never as
    "empty parent" — the difference is a baseline versus an unchecked delta,
    which the evidence card already keeps distinct.
    """
    if not execution_id:
        return None
    path = snapshot_dir(workspace_root, execution_id) / relpath
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
