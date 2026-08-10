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

#: tqdm's completed/total counter, `450/1000`.
_BAR_TOTAL = re.compile(r"\b\d+/(\d+)\b")


def _same_bar(line: str, previous: str) -> bool:
    """Are these two progress lines frames of the *same* bar?

    Keyed on the label before the percentage, because collapsing any two
    adjacent progress-shaped lines merges interleaved bars destructively —
    alternating `Training:` and `Validation:` frames collapsed to one
    `Validation:` line and threw away every frame of both. Reported on PR #117.

    The label alone is not enough: tqdm's default format has none, so two
    unrelated bars both produced an empty prefix and compared equal, losing the
    same state through the no-label case. Also reported on PR #117. The `n/N`
    total is the second key — a bar's total does not change between its own
    frames, and two concurrent bars over different work rarely share one.

    Two unlabeled bars over the same total are genuinely indistinguishable
    here, and collapse. That is the residue; a wrong merge costs intermediate
    frames, never the traceback, which matches no progress shape at all.
    """
    here, there = _PROGRESS_LINE.search(line), _PROGRESS_LINE.search(previous)
    if here is None or there is None:
        return False
    if line[: here.start()] != previous[: there.start()]:
        return False
    return _bar_total(line, here.start()) == _bar_total(previous, there.start())


def _bar_total(line: str, start: int = 0) -> str:
    """The `N` of tqdm's `n/N`, or "" when the line does not carry one.

    Searched from the bar onward, never from the start of the line. `re.search`
    returns the leftmost match, so an epoch marker in a shared label —
    ``"Epoch 3/10 Training: "`` — was read as the total, and two loops over
    genuinely different totals both reported `10` and collapsed into one.
    Reported on PR #117.
    """
    match = _BAR_TOTAL.search(line, start)
    return match.group(1) if match else ""


def stream_text(stream: str | bytes | None) -> str:
    """A captured stream as text, whichever form it arrived in.

    `subprocess.run(text=True)` decodes what it returns, but a `TimeoutExpired`
    it raises carries **bytes** on POSIX: the exception comes from the inner
    `communicate()`, before the decoding step. A caller that interpolates it
    writes a literal ``b'collected 3 items\\n'`` into the log — which looks like a
    record and reads like an escape sequence.

    `errors="replace"` because this is diagnostic output from a process that was
    killed mid-write, so a truncated multi-byte character is expected and losing
    the whole excerpt to it would defeat the purpose. Reported reviewing PR #124.
    """
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream


def stopped_excerpt(
    stdout: str | bytes | None,
    stderr: str | bytes | None,
    *,
    limit: int = _EXCERPT_CHARS,
) -> str:
    """What a process managed to say before it was stopped, from **both** streams.

    `failure_excerpt` takes `stderr or stdout`, which is right for a crash: the
    traceback is on stderr and stdout is noise. A process killed by a timeout has
    no traceback. The tail of stdout says how far it got — which test, which
    epoch, which package — and the tail of stderr says what it was complaining
    about, and either can be the diagnosis.

    Joining the two and excerpting once does not work, because the excerpt keeps
    the *tail*: whichever stream is written last wins, and a stream longer than
    the budget silences the other completely. Two rounds of PR #124 went into
    moving that eviction from one stream to the other. Ordering cannot fix it —
    so each stream gets its own budget and neither can silence the other,
    whatever the volume.

    One function rather than a copy in each capability, because two
    implementations of one idea drifting apart is the defect M20 criterion 2 is
    named after.
    """
    parts = []
    for label, raw in (("stdout", stdout), ("stderr", stderr)):
        text = failure_excerpt(stream_text(raw), "", limit=limit // 2)
        if text:
            parts.append(f"{label}: {text}")
    return "\n".join(parts)


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
        if _same_bar(line, kept[-1] if kept else ""):
            kept[-1] = line
            continue
        kept.append(line)
    text = "\n".join(kept)
    if len(text) <= limit:
        return text
    # The tail, because a traceback names its cause on the last line.
    return "…\n" + text[-limit:]
