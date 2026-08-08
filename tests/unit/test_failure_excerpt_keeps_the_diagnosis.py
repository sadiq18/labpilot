"""A recorded failure must contain the reason it failed.

Measured on rogii 2026-08-08. E-174's stored error was **1523 characters of
progress bar** — `Loading train: 96%|█████████▋|`, repeated — and no diagnosis
at all. `[:1500]` kept the *head* of the output, and a training script's head is
where tqdm lives while its traceback is at the very bottom.

Every training failure was undiagnosable for as long as that held.
"""

from __future__ import annotations

from labpilot.research_engine.execution.capabilities.verification.capability import (
    failure_excerpt,
)

# tqdm redraws one line with \r, so captured output is one line of many frames.
_BAR = "".join(
    f"Loading train: {p}%|{'█' * (p // 10)}| {p}/100 [00:0{p % 9}]\r" for p in range(0, 100)
)

_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "pipeline/train.py", line 88, in main\n'
    "    model.fit(X, y)\n"
    "ValueError: Input contains NaN\n"
)


def test_the_traceback_survives_a_flood_of_progress_bars():
    """The exact shape of E-174, with a real cause appended."""
    excerpt = failure_excerpt(_BAR + "\n" + _TRACEBACK, "")

    assert "ValueError: Input contains NaN" in excerpt


def test_a_progress_bar_collapses_to_one_frame():
    """Each \\r is a redraw of the same line, not new information."""
    excerpt = failure_excerpt(_BAR, "")

    assert excerpt.count("Loading train:") == 1


def test_the_tail_is_kept_not_the_head():
    """A traceback names its cause on the last line."""
    long_noise = "\n".join(f"noise line {i}" for i in range(500))
    excerpt = failure_excerpt(long_noise + "\n" + _TRACEBACK, "")

    assert "ValueError" in excerpt
    assert "noise line 0" not in excerpt


def test_truncation_is_marked():
    """A silently clipped excerpt reads as the whole story."""
    excerpt = failure_excerpt("x" * 5000, "")
    assert excerpt.startswith("…")


def test_stdout_is_used_when_stderr_is_empty():
    assert "boom" in failure_excerpt("", "boom")


def test_stderr_wins_when_both_are_present():
    assert failure_excerpt("real cause", "chatter") == "real cause"


def test_no_output_yields_no_excerpt():
    assert failure_excerpt("", "") == ""
    assert failure_excerpt("   ", "\n") == ""


def test_a_short_failure_is_returned_whole():
    assert failure_excerpt(_TRACEBACK, "") == _TRACEBACK.rstrip()
