"""A CLI runner whose output does not depend on the terminal it ran in.

Rich sizes its tables to the terminal and truncates cells that do not fit, so
`assert "P-001" in result.output` is really an assertion about the width of the
window. At 80 columns it passes; at 40 the same cell renders `P-0…` and it
fails. Eight tests across six files were in that state — found by
`scripts/hostile-test.sh`, which is what the 40-column leg is for.

Pinning the width is the fix `tests/unit/test_tests_do_not_assert_the_machine.py`
prescribes for ambient values generally: inject it, so the assertion says one
thing everywhere. It is not a workaround for the hostile run — a test that pins
the width is portable, which is the property being asked for. What the leg still
catches is the test that *doesn't* pin and reads the ambient width by accident.

If narrow rendering is ever a product requirement, it wants its own test that
asserts the truncation deliberately, not eight that depend on it by accident.
"""

from __future__ import annotations

import sys

from rich.console import Console
from typer.testing import CliRunner

#: Wide enough that no table this CLI renders needs to truncate a cell. Not
#: `None`/auto: the point is that the number is stated rather than inherited.
CLI_COLUMNS = 200


def pin_console_width(width: int = CLI_COLUMNS) -> None:
    """Fix the width of every Rich console the CLI has already built.

    Setting `COLUMNS` in the runner's environment does not work: every
    `labpilot.cli.*` module does `console = Console()` at import, and Rich
    resolves the width once, at construction — by then pytest has long since
    imported the module under whatever `COLUMNS` the shell had. So the width has
    to be assigned on the console objects themselves.
    """
    for name, module in list(sys.modules.items()):
        if not name.startswith("labpilot.cli"):
            continue
        console = getattr(module, "console", None)
        if isinstance(console, Console):
            console.width = width


class _PinnedRunner(CliRunner):
    """A `CliRunner` that re-pins the width on every invoke.

    Per-invoke rather than once, because a command may import a CLI module that
    was not loaded when the runner was built, and that module's console is
    constructed at *its* import — under the ambient terminal.
    """

    def invoke(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        pin_console_width()
        return super().invoke(*args, **kwargs)


def cli_runner(**kwargs) -> CliRunner:
    """A `CliRunner` whose output does not depend on the terminal it ran in."""
    env = {"COLUMNS": str(CLI_COLUMNS), **(kwargs.pop("env", None) or {})}
    return _PinnedRunner(env=env, **kwargs)
