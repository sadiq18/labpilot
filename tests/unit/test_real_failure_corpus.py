"""M20 exit criterion 3: the 2026-08-08 corpus exists, and still gets rejected.

Nine campaigns on 2026-08-08 turned up fifteen defects, **eight of them one
shape** — a gate that tests something easier than it promises, and passes. The
artifacts those gates passed are the best inputs any of these guards will ever
get, and they were scattered inline across nine test files.

This file asserts two things about them.

**They are still described.** An artifact whose provenance is lost is a string
literal: nobody can tell whether it is the thing that happened or something
someone typed, and the difference is the whole point.

**They are still rejected.** Every one is fed to the guard that now owns it. A
regression that reopens any of these defects fails here, against the file that
caused it, rather than against a paraphrase.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from helpers.real_failures import CORPUS, corpus_artifacts, real_failure

from labpilot.research_engine.execution.capabilities._helpers import failure_excerpt
from labpilot.research_engine.execution.capabilities.code_engineering.apply import (
    ApplyError,
    apply_proposal,
    strip_stdlib_dependencies,
)
from labpilot.research_engine.execution.schemas.code_proposal import (
    CodeFileSpec,
    CodeProposal,
)
from labpilot.research_engine.shared.labels import is_record_reference


def test_every_artifact_is_documented():
    """The manifest is the provenance. Without it these are string literals."""
    manifest = (CORPUS / "MANIFEST.md").read_text(encoding="utf-8")

    undocumented = [p.name for p in corpus_artifacts() if p.name not in manifest]

    assert corpus_artifacts(), "the corpus is empty"
    assert not undocumented, f"artifacts with no manifest entry: {undocumented}"


def test_every_artifact_says_where_it_came_from():
    """Dated and sourced, per the exit criterion — a corpus of anonymous bad
    inputs is indistinguishable from invented ones."""
    manifest = (CORPUS / "MANIFEST.md").read_text(encoding="utf-8")

    for path in corpus_artifacts():
        row = next(line for line in manifest.splitlines() if path.name in line)
        assert "2026-" in row, f"{path.name} has no date"
        assert "rogii" in row, f"{path.name} has no source workspace"


def test_a_missing_artifact_is_an_error_not_an_empty_string():
    """An empty string is a *different* bad input, and a guard proven against
    the wrong bad input is the shape this milestone exists to end."""
    with pytest.raises(FileNotFoundError):
        real_failure("never_happened.txt")


# -- each artifact, against the gate that let it through -----------------------


def test_the_truncated_train_py_is_rejected_at_apply():
    """Defects 6 and 8: 624 bytes — a docstring and half a comment. Valid
    Python, so `ast.parse` passed it; `run_smoke_test` passed it too, because
    the file it ran exits 0 by doing nothing."""
    content = real_failure("truncated_train_py.txt")

    with pytest.raises(ApplyError) as caught:
        apply_proposal(
            Path(tempfile.mkdtemp()),
            CodeProposal(
                files=[CodeFileSpec(path="pipeline/train.py", content=content, action="write")]
            ),
        )

    assert "PEP 723" in str(caught.value)


def test_the_truncated_file_still_parses_which_is_the_point():
    """The reason this artifact is worth keeping: the gate that failed was not
    wrong about syntax. It was answering an easier question than it promised."""
    import ast

    ast.parse(real_failure("truncated_train_py.txt"))


def test_the_stdlib_dependency_is_stripped():
    """Defect 11: codegen declared `glob`, and uv refused all six dependencies —
    the run never started, so no other gate got a chance to see it."""
    content = real_failure("stdlib_dependency_block.txt")

    _, dropped = strip_stdlib_dependencies(content)

    assert dropped == ["glob"]


def test_the_tqdm_flood_keeps_the_diagnosis():
    """Defect 9: the stored error was 1523 characters of `Loading train: 96%`
    and no traceback. `error[:1500]` kept the head, and the head is the bar."""
    excerpt = failure_excerpt(real_failure("tqdm_flood_stderr.txt"), "", limit=1500)

    assert "could not convert string to float" in excerpt
    assert excerpt.count("Loading train") == 1


def test_the_record_reference_is_not_a_technique():
    """Defect 1: `hyp:H-010` reached six agents' system prompts on every run,
    as though it were a technique to avoid."""
    assert is_record_reference(real_failure("record_reference_technique.txt").strip())
