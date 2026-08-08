"""The aider invocation, pinned where the defaults would cost us.

Every flag here was chosen against a measurement or a known failure, and none of
them announces itself when it regresses — a wrong edit format still succeeds and
simply costs more, and a missing `--no-stream` still returns an answer while
quietly unmetering it.
"""

from __future__ import annotations

import pytest

from labpilot.research_engine.execution.delta.aider_agent import (
    CODEGEN_ROLE,
    EDIT_FORMAT,
    _aider_command,
)


@pytest.fixture
def cmd() -> list[str]:
    return _aider_command("http://127.0.0.1:5555/v1", CODEGEN_ROLE, ["pipeline/train.py"], "do it")


def _flag(cmd: list[str], name: str) -> str:
    return cmd[cmd.index(name) + 1]


def test_the_edit_format_is_pinned_to_diff(cmd):
    """Unset, aider picks `whole` for a model name it has no data for, and
    re-emits the entire file — the waste M19 exists to remove. Measured
    2026-08-09: diff +20/-7 at 8.0k tokens, whole +23/-7 at 9.1k."""
    assert _flag(cmd, "--edit-format") == "diff"
    assert EDIT_FORMAT == "diff"


def test_the_model_names_a_role_not_a_vendor(cmd):
    """`labpilot/<role>` is the only form the proxy accepts; naming a provider
    model would bypass role selection entirely."""
    assert _flag(cmd, "--model") == "openai/labpilot/codegen"


def test_it_points_at_the_given_proxy(cmd):
    assert _flag(cmd, "--openai-api-base") == "http://127.0.0.1:5555/v1"


def test_streaming_is_off(cmd):
    """An OpenAI-compatible stream omits `usage` unless the provider honours
    `stream_options`, and an unmetered call defeats the ledger the proxy exists
    to feed."""
    assert "--no-stream" in cmd


def test_git_is_off(cmd):
    """The workspace copy is scratch. Letting aider init or commit in it would
    put a second VCS inside a tree labpilot already versions."""
    assert "--no-git" in cmd
    assert "--no-auto-commits" in cmd


def test_it_never_prompts(cmd):
    """This runs as a subprocess with no terminal; a prompt is a hang."""
    assert "--yes-always" in cmd


def test_the_instruction_and_targets_are_passed(cmd):
    assert _flag(cmd, "--message") == "do it"
    assert cmd[-1] == "pipeline/train.py"


def test_every_edit_target_is_named():
    cmd = _aider_command("http://x/v1", "codegen", ["pipeline/train.py", "pipeline/util.py"], "go")
    assert cmd[-2:] == ["pipeline/train.py", "pipeline/util.py"]


def test_the_argv_is_never_shell_parsed(cmd):
    """Built as a list so an instruction containing quotes or semicolons is
    data, not syntax."""
    hostile = _aider_command("http://x/v1", "codegen", ["a.py"], 'rm -rf /; echo "pwned"')
    assert _flag(hostile, "--message") == 'rm -rf /; echo "pwned"'
