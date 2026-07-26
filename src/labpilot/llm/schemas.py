"""Shared Pydantic types for the LLM layer (routing / cache keys)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedRoute:
    """Concrete provider+model selection for one generate() call."""

    provider: str
    model: str
    temperature: float
    task: str
