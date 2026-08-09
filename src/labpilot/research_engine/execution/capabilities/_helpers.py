"""Shared helpers for capability evidence."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import TaskEvidence


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def evidence(
    context: TaskContext,
    *,
    capability: str,
    passed: bool,
    summary: str,
    checks: list[str] | None = None,
    paths: list[str] | None = None,
    error: str | None = None,
    metrics: dict | None = None,
    metadata: dict | None = None,
) -> TaskEvidence:
    return TaskEvidence(
        task_id=context.task.id,
        execution_id=context.execution.id,
        capability=capability,
        passed=passed,
        summary=summary,
        checks=checks or [],
        paths=paths or [],
        error=error,
        metrics=metrics or {},
        metadata=metadata or {},
    )


def is_dry_run(context: TaskContext) -> bool:
    return bool(context.constraints.get("dry_run", False))


def allow_upload(context: TaskContext) -> bool:
    return bool(context.constraints.get("allow_upload", False))


_EXCERPT_CHARS = 1500


#: tqdm's own shape: a percentage immediately before the bar, or a rate suffix.
#: Deliberately not "consecutive lines that look alike" — a traceback's
#: ``  File "…"`` lines look alike and every one of them matters.
_PROGRESS_LINE = re.compile(r"\d+%\|| \d+\.?\d*(it|s)/(s|it)\]")


def failure_excerpt(stderr: str, stdout: str, *, limit: int = _EXCERPT_CHARS) -> str:
    """The part of a failed run worth reading: the end, minus progress bars.

    Taking `[:limit]` kept the *head*, and a training script's head is where
    tqdm lives while its traceback is at the very bottom. Measured on rogii
    2026-08-08: E-174's stored error was 1523 characters of
    ``Loading train: 96%|█████████▋|`` and contained no diagnosis at all — the
    run failed for a reason nothing recorded.

    tqdm redraws one line by emitting ``\\r``, so each carriage return is a
    frame of the same bar. Keeping only the text after the last ``\\r`` collapses
    a bar to its final state instead of preserving every frame, which is what
    filled the budget.
    """
    raw = stderr.strip() or stdout.strip()
    if not raw:
        return ""
    # Split on newlines only. `str.splitlines()` also breaks on `\r`, which
    # would hand every tqdm frame back as its own line and defeat the collapse
    # entirely — the first version of this function did exactly that.
    collapsed = "\n".join(line.rsplit("\r", 1)[-1].rstrip() for line in raw.split("\n"))
    # Drop frames that survived collapsing but still say nothing.
    kept: list[str] = []
    for line in collapsed.splitlines():
        if not line.strip():
            continue
        if _PROGRESS_LINE.search(line) and kept and _PROGRESS_LINE.search(kept[-1]):
            kept[-1] = line
            continue
        kept.append(line)
    text = "\n".join(kept)
    if len(text) <= limit:
        return text
    # The tail, because a traceback names its cause on the last line.
    return "…\n" + text[-limit:]
