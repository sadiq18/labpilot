"""Structured git results for agents."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class GitStatus(BaseModel):
    """Working tree status summary."""

    branch: str | None = None
    clean: bool = True
    staged: list[str] = Field(default_factory=list)
    unstaged: list[str] = Field(default_factory=list)
    untracked: list[str] = Field(default_factory=list)


class DiffSummary(BaseModel):
    """Compact diff summary (paths + optional text)."""

    files_changed: list[str] = Field(default_factory=list)
    text: str = ""


class CommitSnapshot(BaseModel):
    """Agent-facing commit result."""

    commit: str
    short: str = ""
    message: str = ""
    files_changed: list[str] = Field(default_factory=list)
    branch: str | None = None

    @model_validator(mode="after")
    def _fill_short(self) -> CommitSnapshot:
        if not self.short and self.commit:
            self.short = self.commit[:7]
        return self


class BranchInfo(BaseModel):
    name: str
    commit: str | None = None


class GitLogEntry(BaseModel):
    commit: str
    short: str = ""
    message: str = ""
    author: str = ""

    @model_validator(mode="after")
    def _fill_short(self) -> GitLogEntry:
        if not self.short and self.commit:
            self.short = self.commit[:7]
        return self
