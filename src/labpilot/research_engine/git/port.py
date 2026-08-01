"""GitTool port — agents depend on this, not GitPython or CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from labpilot.research_engine.git.models import (
    BranchInfo,
    CommitSnapshot,
    DiffSummary,
    GitLogEntry,
    GitStatus,
)


@runtime_checkable
class GitTool(Protocol):
    """Structured git operations for Implementation / Experiment specialists."""

    root: Path

    def status(self) -> GitStatus:
        ...

    def diff(self, *, staged: bool = False) -> DiffSummary:
        ...

    def commit(
        self,
        message: str,
        *,
        paths: Sequence[str] | None = None,
    ) -> CommitSnapshot | None:
        """Stage ``paths`` (default: code roots) and commit. None if empty commit."""
        ...

    def checkout(self, ref: str, *, paths: Sequence[str] | None = None) -> None:
        """Checkout branch/commit, or restore ``paths`` from ``ref`` when paths set."""
        ...

    def create_branch(self, name: str, *, checkout: bool = True) -> BranchInfo:
        ...

    def log(self, *, max_count: int = 20) -> list[GitLogEntry]:
        ...

    def get_commit(self, commit_id: str = "HEAD") -> CommitSnapshot | None:
        ...

    def execute(self, *git_args: str) -> str:
        """CLI fallback for advanced ops (bisect, etc.). Returns combined output."""
        ...
