"""Typed artifact for Code Engineering micro-agent proposals."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CodeFileSpec(BaseModel):
    """One file the agent wants written under the competition workspace."""

    path: str
    content: str
    action: Literal["write"] = "write"


class CodeProposal(BaseModel):
    """LLM/rule_engine output — deterministic code applies this to disk."""

    summary: str = ""
    rationale: str = ""
    files: list[CodeFileSpec] = Field(default_factory=list)
