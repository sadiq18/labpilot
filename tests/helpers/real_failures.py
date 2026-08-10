"""Access to `tests/fixtures/real_failures/` — M20 exit criterion 3.

A guard is proven by the failure it rejects, and the failure has to be the real
one. `15-gates-must-fail.md` names the trap directly: a hand-written "truncated
file" would have had no `# /// script` block, sailed through the check, and
taught nothing. The one that actually occurred is truncated *inside* the block,
which is why `ast.parse` accepted it.

These artifacts were inline across nine test files before this module, one copy
per guard, each free to drift from what really happened.
"""

from __future__ import annotations

from pathlib import Path

CORPUS = Path(__file__).resolve().parent.parent / "fixtures" / "real_failures"


def real_failure(name: str) -> str:
    """One artifact, verbatim. Raises if it is missing rather than returning "".

    An empty string would be a *different* bad input, and a guard proven against
    the wrong bad input is the shape M20 exists to end.
    """
    path = CORPUS / name
    if not path.is_file():
        available = ", ".join(sorted(p.name for p in CORPUS.glob("*.txt")))
        raise FileNotFoundError(f"no real-failure artifact {name!r}. Have: {available}")
    return path.read_text(encoding="utf-8")


def corpus_artifacts() -> list[Path]:
    """Every artifact in the corpus, manifest excluded."""
    return sorted(CORPUS.glob("*.txt"))
