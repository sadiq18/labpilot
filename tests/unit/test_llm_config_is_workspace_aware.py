"""Resolving an LLM client must read the *workspace* config, not the package default.

`load_config()` layers the package `configs/default.yaml` and the environment.
Routing is deliberately empty there, so a caller that pairs it with
`resolve_llm_client` gets `routing.providers == []`, `build_gateway` returns
None, and the call falls through to the legacy provider pin — silently, with a
working client to show for it.

Measured on rogii 2026-08-12: the workspace named fourteen routable endpoints in
`configs/default.yaml` and every campaign ran on `ollama` anyway, because
`analyze_competition` resolved its client this way. `diagnostics.py` already used
`load_config_for_cwd` and said why in a comment; the three analyzers did not.

Discovery rather than a list of the three, because the next caller to get this
wrong is the one nobody thought to add here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "labpilot"

#: Loaders that see only the package default. `load_config(path)` with an
#: explicit path is fine — the caller has already decided which file to read.
PACKAGE_ONLY_LOADERS = {"load_config"}

#: Entry points that turn an `LLMConfig` into a client or a router. Both consult
#: `routing`, so both need the workspace's copy of it.
NEEDS_ROUTING = {"resolve_llm_client", "build_gateway"}


def _called_names(node: ast.AST) -> list[tuple[str, ast.Call]]:
    """`(name, call)` for every call in `node`, by its bare function name."""
    found = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if name:
            found.append((name, child))
    return found


def _offenders(tree: ast.AST) -> list[str]:
    """Functions that resolve a client from a package-default config load."""
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        calls = _called_names(node)
        if not any(name in NEEDS_ROUTING for name, _ in calls):
            continue
        for name, call in calls:
            # Only the no-argument form. `load_config(workspace.config_path)`
            # has already chosen a file and is not what this test is about.
            if name in PACKAGE_ONLY_LOADERS and not call.args and not call.keywords:
                bad.append(f"{node.name} (line {call.lineno})")
    return bad


@pytest.mark.parametrize(
    "source", sorted(SRC.rglob("*.py")), ids=lambda p: str(p.relative_to(SRC))
)
def test_a_client_is_never_resolved_from_the_package_default(source: Path) -> None:
    offenders = _offenders(ast.parse(source.read_text(encoding="utf-8")))

    assert not offenders, (
        f"{source.relative_to(SRC)}: {', '.join(offenders)} resolves an LLM client from "
        "`load_config()`, which cannot see the workspace's `llm.routing`. The client comes "
        "back working and wrong — the legacy provider pin instead of the router. "
        "Use `load_config_for_cwd()[0]`."
    )


def test_the_rule_catches_the_shape_it_was_written_for() -> None:
    """The detector against the exact code that shipped the bug.

    Without this, a rule that matches nothing passes every file above and reads
    as evidence the codebase is clean.
    """
    shipped = ast.parse(
        "def _maybe_attach_llm_client(self):\n"
        "    self.llm_client = resolve_llm_client(load_config().llm)\n"
    )
    fixed = ast.parse(
        "def _maybe_attach_llm_client(self):\n"
        "    self.llm_client = resolve_llm_client(load_config_for_cwd()[0].llm)\n"
    )

    assert _offenders(shipped), "the detector no longer sees the bug it was written for"
    assert not _offenders(fixed), "the detector flags the fix"
