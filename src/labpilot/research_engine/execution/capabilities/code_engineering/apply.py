"""Deterministic apply of a CodeProposal under an allow-list."""

from __future__ import annotations

import ast
import logging
import re
import sys
import tomllib
from pathlib import Path

from labpilot.research_engine.execution.schemas.code_proposal import CodeProposal

#: Modules whose import name differs from the distribution that provides them.
#:
#: Deliberately short. A long curated list answering an open-world question is
#: the pattern this codebase has rejected repeatedly — it goes stale silently
#: and starts refusing correct code. These are the ones a `train.py` actually
#: reaches for; anything else is compared by name, which is right far more
#: often than not, and a wrong guess here costs a re-ask rather than a run.
IMPORT_ALIASES: dict[str, str] = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "skimage": "scikit-image",
}

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


def _uncomment(line: str) -> str:
    """`  # dependencies = [...]` -> `dependencies = [...]`.

    Leading whitespace before the `#` is legal and a model writes it. The first
    version matched a literal `"# "` at position zero and fell through to
    `lstrip("#")` otherwise, which left the space in front of the marker and
    made the whole block unparseable — so a script that declared its imports
    correctly was rejected for not declaring them, the opposite of what this
    check exists to catch. Reported on PR #118.
    """
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return stripped
    return stripped[1:].removeprefix(" ")


def _normalise_distribution(name: str) -> str:
    """PEP 503 normalisation: `scikit_learn`, `Scikit.Learn` -> `scikit-learn`.

    Comparing lowercased names alone rejected `scikit_learn` declared against a
    `sklearn` import — a correct script refused over a separator. Reported on
    PR #118. PEP 503 is the packaging standard for this, so it is not a guess.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_dependencies(content: str) -> set[str] | None:
    """Distribution names in the PEP 723 block, or None when there is none.

    Parsed as TOML, which is what PEP 723 says the block is, rather than by the
    line regex next door. That regex reads one quoted spec per line and misses
    `# dependencies = ["lightgbm>=4.0"]` — the inline form, which is both legal
    and what a model writes about half the time. A completeness check that
    cannot see the declarations would report every such script as missing all
    of them.
    """
    lines = content.splitlines()
    opening = next((i for i, line in enumerate(lines) if _PEP723_OPEN.match(line)), None)
    if opening is None:
        return None
    closing = next(
        (i for i in range(opening + 1, len(lines)) if _PEP723_CLOSE.match(lines[i])),
        None,
    )
    if closing is None:
        return None
    body = "\n".join(_uncomment(line) for line in lines[opening + 1 : closing])
    try:
        declared = tomllib.loads(body).get("dependencies") or []
    except tomllib.TOMLDecodeError:
        # Malformed metadata is uv's complaint to make, not a reason to refuse
        # a script over an import it may well have declared.
        return None
    if not isinstance(declared, list):
        return None
    return {_normalise_distribution(_distribution_name(str(spec))) for spec in declared}


def _imported_modules(tree: ast.Module) -> set[str]:
    """Top-level module names imported anywhere, except under `try:`."""
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for statement in node.body:
                for inner in ast.walk(statement):
                    guarded.add(id(inner))
    found: set[str] = set()
    for node in ast.walk(tree):
        if id(node) in guarded:
            continue
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def _check_dependencies_are_complete(rel: str, content: str, tree: ast.Module) -> None:
    """Every third-party module the script imports must be declared.

    `uv run --script` builds the environment from the PEP 723 block *only*, so
    an import the block does not name is a `ModuleNotFoundError` at run time —
    one campaign step spent to learn that codegen forgot a line of metadata.
    PR #102 fixed this once for the Jinja pack, where a test checked the same
    property over `.j2` source; M19 §2 deleted the pack and the test with it,
    and every `train.py` is now model-written, which is where the mistake is
    *more* likely, not less. Reported on PR #118.

    Asked only when a block exists at all — a script without one is not run by
    `uv run --script`. Imports at *any* depth count: scanning `tree.body` alone
    missed the deferred-import idiom (`def main(): import xgboost`), which is
    an undeclared dependency exactly like a top-level one and fails the same
    way. Reported on PR #118. Imports under `try:` are excluded, because that
    is how optional dependencies are written and their absence is handled by
    the code itself.

    `IMPORT_ALIASES` covers the handful of modules whose import name differs
    from their distribution name; anything else is compared after PEP 503
    normalisation, which is right far more often than not and errs toward
    silence when it is not.
    """
    declared = _declared_dependencies(content)
    if declared is None:
        return
    missing = sorted(
        module
        for module in _imported_modules(tree)
        if module not in sys.stdlib_module_names
        and module != "labpilot"
        and _normalise_distribution(IMPORT_ALIASES.get(module, module)) not in declared
    )
    if missing:
        raise ApplyError(
            f"{rel} imports {', '.join(missing)} but its PEP 723 block does not "
            "declare " + ("it" if len(missing) == 1 else "them") + ". "
            "`uv run --script` builds the environment from that block alone, so "
            "the run would fail with ModuleNotFoundError."
        )


def _imports_labpilot(tree: ast.Module) -> str:
    """The first real `labpilot` import, or "" — read from the AST.

    A regex over raw text also matches inside docstrings and comments, so a
    generated script whose module docstring *documents* the constraint
    ("never `import labpilot` in a declaring script") would be rejected for
    obeying it. Reported on PR #118.

    Takes the parsed tree rather than the source: the first version re-parsed
    the same string the caller had just parsed, which doubled the AST cost on
    the code-writing path while its own docstring claimed the tree was already
    available. Reported on PR #118 as well.
    """
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


def _check_standalone_script(rel: str, content: str, tree: ast.Module) -> None:
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
    found = _imports_labpilot(tree)
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


def _normalise(path: str) -> str:
    """`./pipeline/x.py` -> `pipeline/x.py`, leaving every other segment alone.

    `lstrip("./")` strips a *character set*, not a prefix, so
    `"../pipeline/evil.py"` came back as `"pipeline/evil.py"` — the traversal
    erased before `_is_allowed` could see the `..` it exists to reject. The
    stripped result still lands under an allowed root, so nothing escaped the
    workspace; what was lost was the error. A model that wrote `..` meaning one
    level up got a silent write somewhere else instead of a rejection telling
    it so. Reported on PR #118.
    """
    norm = path.replace("\\", "/")
    while norm.startswith("./"):
        norm = norm[2:]
    return norm


def _missing_parents(target: Path) -> list[Path]:
    """Directories that would have to be created for `target`, deepest first."""
    missing: list[Path] = []
    parent = target.parent
    while not parent.exists() and parent != parent.parent:
        missing.append(parent)
        parent = parent.parent
    return missing


def _roll_back(
    attempted: list[Path],
    previous: dict[Path, bytes | None],
    created_dirs: list[Path],
) -> None:
    """Put the tree back, best effort, without ever raising.

    `attempted`, not `written`. The file that raised is the one most likely to
    need restoring and was the one never restored: it is appended to `written`
    only *after* `write_text` returns, so the rollback skipped it. That is not
    a corner — it is what a disk-full write looks like. `open(...,"w")` succeeds
    and truncates, then `write()` fails, so the original content is already
    gone when the exception arrives. Reported on PR #118, with the file
    destroyed and the error saying "Nothing was applied."

    Every restore is isolated: the condition that broke the write — a full
    disk, a revoked permission — is usually still true, so a second `OSError`
    is likely, and letting it out used to abandon every file after it and
    replace the real diagnosis with the rollback's own. Directories this apply
    created are removed too, since "nothing was applied" is not true of a tree
    left holding new empty ones. Reported on PR #118.
    """
    for target in attempted:
        original = previous.get(target)
        try:
            if original is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(original)
        except OSError as exc:  # noqa: PERF203 - each file is restored or reported
            logger.warning("could not roll back %s: %s", target, exc)
    for directory in sorted(set(created_dirs), key=lambda p: len(p.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            # Not empty, or gone already. Either way there is nothing to undo.
            pass


def _is_allowed(rel: str) -> bool:
    norm = _normalise(rel)
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
        rel = _normalise(spec.path)
        if not _is_allowed(rel) or not any(
            rel == root or rel.startswith(f"{root}/") for root in allowed_roots
        ):
            raise ApplyError(f"path not allowed: {spec.path}")
        if not spec.content.strip():
            raise ApplyError(f"empty content for {spec.path}")

        content = spec.content
        if rel.endswith(".py"):
            try:
                tree = ast.parse(content)
            except SyntaxError as exc:
                raise ApplyError(f"syntax error in {rel}: {exc}") from exc
            # Syntax is necessary and not sufficient — both checks below pass
            # `ast.parse` and still cannot run.
            _check_dependency_block(rel, content)
            # Before completeness: `labpilot` is undeclared *and* undeclarable,
            # and `_check_standalone_script` says why in terms the model can
            # act on. Completeness would report it as a missing dependency and
            # send the next attempt to add it to the block, which cannot work.
            _check_standalone_script(rel, content, tree)
            _check_dependencies_are_complete(rel, content, tree)
            _check_not_truncated(rel, content)
            content, dropped = strip_stdlib_dependencies(content)
            if dropped:
                logger.info(
                    "Dropped stdlib module(s) from %s dependencies: %s",
                    rel,
                    ", ".join(dropped),
                )
        staged.append((workspace_root / rel, content))

    # Validation guarantees nothing is written when a *later* file is refused.
    # It said nothing about the write loop itself: a failure on file N left
    # files 1..N-1 on disk, which is precisely the "neither the parent nor the
    # proposal" state this function exists to prevent — the docstring above
    # claimed the property and the code held it only for one of the two ways to
    # lose it. Reported on PR #118. So the prior bytes are kept and put back.
    written: list[Path] = []
    attempted: list[Path] = []
    created_dirs: list[Path] = []
    previous: dict[Path, bytes | None] = {}
    failed: Path | None = None
    try:
        # Snapshotting is inside the guard, not before it. An unreadable
        # pre-existing file used to raise a bare `PermissionError` out of a
        # module whose every other rejection is an `ApplyError`. Reported on
        # PR #118.
        for target, _ in staged:
            previous[target] = target.read_bytes() if target.is_file() else None
        for target, content in staged:
            failed = target
            attempted.append(target)
            created_dirs.extend(_missing_parents(target))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            written.append(target)
    except OSError as exc:
        _roll_back(attempted, previous, created_dirs)
        raise ApplyError(f"could not write {failed}: {exc}. Nothing was applied.") from exc
    return written
