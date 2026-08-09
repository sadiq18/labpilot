"""Deterministic apply of a CodeProposal under an allow-list."""

from __future__ import annotations

import ast
import logging
import re
import sys
from pathlib import Path

from labpilot.research_engine.execution.schemas.code_proposal import CodeProposal

ALLOWED_ROOTS = ("pipeline", "src", "configs", "tests", "artifacts")

#: The training entry point. `read_code`'s skill states the contract: *"Always
#: end `pipeline/train.py` with exact ``if __name__ == "__main__":`` +
#: ``main()``"*. Checking it here turns a documented promise into a gate.
TRAIN_RELPATH = "pipeline/train.py"

_MAIN_GUARD = re.compile(r"^if\s+__name__\s*==\s*[\"']__main__[\"']\s*:", re.MULTILINE)
_PEP723_OPEN = re.compile(r"^#\s*///\s*script\s*$", re.MULTILINE)
_PEP723_CLOSE = re.compile(r"^#\s*///\s*$", re.MULTILINE)


logger = logging.getLogger(__name__)

#: One dependency line inside a PEP 723 block: `#   "lightgbm>=4.0",`
_DEP_LINE = re.compile(r"""^#\s*(["'])(?P<spec>[^"']+)\1\s*,?\s*$""")


class ApplyError(ValueError):
    """Proposal rejected (path escape, empty, syntax, truncation)."""


def _distribution_name(spec: str) -> str:
    """`lightgbm>=4.0` -> `lightgbm`; `pkg[extra]` -> `pkg`."""
    return re.split(r"[<>=!~;\[\s]", spec.strip(), maxsplit=1)[0].strip()


def strip_stdlib_dependencies(content: str) -> tuple[str, list[str]]:
    """Drop stdlib modules from a PEP 723 block. Returns (content, dropped).

    `uv` resolves declared dependencies against PyPI, so one stdlib name makes
    the *whole* set unsatisfiable — measured on rogii 2026-08-08, codegen
    declared `glob` alongside four real packages and uv refused all five:
    *"Because glob was not found in the package registry … your requirements
    are unsatisfiable."* The run never started.

    Repaired rather than rejected, for the reason `code_proposal.py` already
    gives about coercing `null`: the cost of strictness here is losing an
    experiment over a line of metadata, while the cost of leniency is a
    correct file arriving in a slightly different shape. The edit is
    mechanical and total — a stdlib module is never a PyPI dependency.

    `sys.stdlib_module_names` is the authority, so this is **not** the
    curated-set-answering-an-open-world-question pattern rejected four times
    elsewhere: Python itself owns the answer, and it updates with the runtime.
    """
    lines = content.splitlines(keepends=True)
    opening = next(
        (i for i, line in enumerate(lines) if _PEP723_OPEN.match(line.rstrip("\n"))),
        None,
    )
    if opening is None:
        return content, []
    closing = next(
        (i for i in range(opening + 1, len(lines)) if _PEP723_CLOSE.match(lines[i].rstrip("\n"))),
        None,
    )
    if closing is None:
        # Unterminated — `_check_dependency_block` owns that failure, and
        # editing a block whose extent is unknown would guess at its end.
        return content, []

    dropped: list[str] = []
    kept: list[str] = []
    for line in lines[opening + 1 : closing]:
        match = _DEP_LINE.match(line.rstrip("\n"))
        if match and _distribution_name(match.group("spec")) in sys.stdlib_module_names:
            dropped.append(_distribution_name(match.group("spec")))
            continue
        kept.append(line)
    if not dropped:
        return content, []
    return "".join(lines[: opening + 1] + kept + lines[closing:]), dropped


def _check_dependency_block(rel: str, content: str) -> None:
    """An opened PEP 723 block must be closed.

    `uv` refuses the whole script otherwise — *"An opening tag (`# /// script`)
    was found without a closing tag"* — so an unterminated block is not a
    partial win, it is a run that cannot start. Discovered at run time it costs
    a campaign step; discovered here it is a rejected proposal and a re-ask.
    """
    if not _PEP723_OPEN.search(content):
        return
    opening = _PEP723_OPEN.search(content)
    assert opening is not None
    if not _PEP723_CLOSE.search(content, opening.end()):
        raise ApplyError(
            f"unterminated PEP 723 block in {rel}: `# /// script` was opened and "
            "never closed with `# ///`. uv rejects the whole script, so the run "
            "cannot start."
        )


def _imports_labpilot(content: str) -> str:
    """The first real `labpilot` import, or "" — read from the AST.

    A regex over raw text also matches inside docstrings and comments, so a
    generated script whose module docstring *documents* the constraint
    ("never `import labpilot` in a declaring script") would be rejected for
    obeying it. Reported on PR #118, and the tree is already parsed two lines
    earlier.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:  # pragma: no cover - the caller parses first
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "labpilot" or alias.name.startswith("labpilot."):
                    return f"import {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "labpilot" or module.startswith("labpilot."):
                return f"from {module} import ..."
    return ""


def _check_standalone_script(rel: str, content: str) -> None:
    """A script that declares its own dependencies cannot import labpilot.

    `uv run --script` builds an ephemeral environment from the PEP 723 block,
    and labpilot is not in it — so the two together are a `ModuleNotFoundError`
    at run time, one campaign step spent to learn it.

    This invariant used to be enforced over the Jinja template pack, where it
    was caught in review on PR #102: PEP 723 blocks had been added to two
    templates that both did `from labpilot.research_engine.execution.metrics
    import compute_metric`. M19 §2 deleted the pack, and the rule moved here —
    to the gate every proposal passes through — because it was never really
    about templates. Generated code is where it applies now.
    """
    if not _PEP723_OPEN.search(content):
        return
    found = _imports_labpilot(content)
    if not found:
        return
    raise ApplyError(
        f"{rel} declares PEP 723 dependencies and also imports labpilot "
        f"({found!r}). `uv run --script` runs it in an "
        "ephemeral environment where labpilot is absent, so the script cannot "
        "start. Declare dependencies and stand alone, or use labpilot's "
        "environment — never both."
    )


def _check_not_truncated(rel: str, content: str) -> None:
    """`ast.parse` cannot see a file that was cut off inside its comments.

    Measured on rogii 2026-08-08: codegen returned 624 bytes — a docstring and
    half a `# requires-python = ` line — and the syntax gate passed it, because
    a docstring followed by comments is valid Python that simply does nothing.
    The training run then failed on the unterminated dependency block, and the
    campaign lost seven executions to a file with no code in it.

    Only `train.py` is checked. Helper modules legitimately have no entry point,
    and requiring one would reject correct code to catch a rare truncation.
    """
    if rel != TRAIN_RELPATH:
        return
    if not _MAIN_GUARD.search(content):
        raise ApplyError(
            f'{rel} has no `if __name__ == "__main__":` guard, so it defines no '
            "entry point to run. The usual cause is a truncated response: the "
            "file parses because what survived was a docstring and comments."
        )


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
    """Write proposal files; return written paths. Validates Python syntax.

    **Validate every file, then write every file.** M19 §2's second exit
    criterion is that the workspace is untouched when a proposal is rejected,
    and a single loop that validated and wrote each file in turn could not meet
    it: a proposal refused on its third file had already written the first two,
    leaving a tree that is neither the parent nor the proposal.

    That state is worse than either. The next experiment's parent is whatever
    the half-apply left, so a rejected proposal silently becomes the baseline
    for the run after it — and the delta checks then compare against a file
    nobody proposed. Measured on rogii 2026-08-09 in the neighbouring case: a
    *failed* run's leftover edit made the next attempt at the same hypothesis
    look already implemented, and it was retired for work that had never run.

    Retries are deliberately unaffected. A rejected proposal writes nothing; a
    proposal that applied and then failed downstream keeps its files, which is
    what the retry loop reads and repairs.
    """
    if not proposal.files:
        raise ApplyError("CodeProposal has no files")

    staged: list[tuple[Path, str]] = []
    for spec in proposal.files:
        rel = spec.path.replace("\\", "/").lstrip("./")
        if not _is_allowed(rel) or not any(
            rel == root or rel.startswith(f"{root}/") for root in allowed_roots
        ):
            raise ApplyError(f"path not allowed: {spec.path}")
        if not spec.content.strip():
            raise ApplyError(f"empty content for {spec.path}")

        content = spec.content
        if rel.endswith(".py"):
            try:
                ast.parse(content)
            except SyntaxError as exc:
                raise ApplyError(f"syntax error in {rel}: {exc}") from exc
            # Syntax is necessary and not sufficient — both checks below pass
            # `ast.parse` and still cannot run.
            _check_dependency_block(rel, content)
            _check_standalone_script(rel, content)
            _check_not_truncated(rel, content)
            content, dropped = strip_stdlib_dependencies(content)
            if dropped:
                logger.info(
                    "Dropped stdlib module(s) from %s dependencies: %s",
                    rel,
                    ", ".join(dropped),
                )
        staged.append((workspace_root / rel, content))

    written: list[Path] = []
    for target, content in staged:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(target)
    return written
