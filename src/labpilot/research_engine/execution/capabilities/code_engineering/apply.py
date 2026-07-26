"""Deterministic apply of a CodeProposal under an allow-list."""

from __future__ import annotations

import ast
from pathlib import Path

from labpilot.research_engine.execution.schemas.code_proposal import CodeProposal

ALLOWED_ROOTS = ("pipeline", "src", "configs", "tests", "artifacts")


class ApplyError(ValueError):
    """Proposal rejected (path escape, empty, syntax)."""


def _is_allowed(rel: str) -> bool:
    norm = rel.replace("\\", "/").lstrip("./")
    if not norm or norm.startswith("/") or ".." in Path(norm).parts:
        return False
    return any(norm == root or norm.startswith(f"{root}/") for root in ALLOWED_ROOTS)


def apply_proposal(
    workspace_root: Path,
    proposal: CodeProposal,
    *,
    allowed_roots: tuple[str, ...] = ALLOWED_ROOTS,
) -> list[Path]:
    """Write proposal files; return written paths. Validates Python syntax."""
    written: list[Path] = []
    if not proposal.files:
        raise ApplyError("CodeProposal has no files")

    for spec in proposal.files:
        rel = spec.path.replace("\\", "/").lstrip("./")
        if not _is_allowed(rel) or not any(
            rel == root or rel.startswith(f"{root}/") for root in allowed_roots
        ):
            raise ApplyError(f"path not allowed: {spec.path}")
        if not spec.content.strip():
            raise ApplyError(f"empty content for {spec.path}")

        target = workspace_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".py"):
            try:
                ast.parse(spec.content)
            except SyntaxError as exc:
                raise ApplyError(f"syntax error in {rel}: {exc}") from exc
        target.write_text(spec.content, encoding="utf-8")
        written.append(target)
    return written
