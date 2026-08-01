"""GitTool — structured git for specialists (GitPython + CLI fallback)."""

from __future__ import annotations

from labpilot.research_engine.git.models import (
    BranchInfo,
    CommitSnapshot,
    DiffSummary,
    GitLogEntry,
    GitStatus,
)
from labpilot.research_engine.git.port import GitTool
from labpilot.research_engine.git.python_backend import (
    DEFAULT_CODE_PATHS,
    GitPythonTool,
    open_git_tool,
)

CODE_PATHS = DEFAULT_CODE_PATHS

__all__ = [
    "CODE_PATHS",
    "DEFAULT_CODE_PATHS",
    "BranchInfo",
    "CommitSnapshot",
    "DiffSummary",
    "GitLogEntry",
    "GitPythonTool",
    "GitStatus",
    "GitTool",
    "open_git_tool",
]
