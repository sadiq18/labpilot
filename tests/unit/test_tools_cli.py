"""`research tools list` — M15 exit criterion 2.

*"`research tools` prints the inventory with capability status."* The point
is that an operator can tell which tools can actually change an outcome
without reading source, so these assert on what reaches stdout, not on the
descriptors (which `test_tool_contracts.py` already covers).
"""

from __future__ import annotations

from typer.testing import CliRunner

from labpilot.cli import main as cli_main
from labpilot.research_engine.tools.catalog import default_tool_descriptors

runner = CliRunner()


def _list_output() -> str:
    result = runner.invoke(cli_main.app, ["tools", "list"])
    assert result.exit_code == 0, result.stdout
    return result.stdout


def test_every_tool_appears_with_its_status() -> None:
    output = _list_output()
    # Rich wraps long cells, so compare on a whitespace-collapsed copy.
    flat = " ".join(output.split())
    for descriptor in default_tool_descriptors():
        assert descriptor.name in flat, f"{descriptor.name} missing from `tools list`"
    for status in ("real", "partial", "fixed"):
        assert status in flat, f"no tool rendered with status {status!r}"


def test_varies_by_is_visible_for_a_real_tool() -> None:
    """The inventory must say *what* varies, not just that something does.

    `hypothesis_id` is `generate_plan`'s declared input; without this column
    an operator reads "real" and still cannot tell which knob to turn.
    """
    flat = " ".join(_list_output().split())
    assert "hypothesis_id" in flat


def test_the_legend_explains_the_three_statuses() -> None:
    flat = " ".join(_list_output().split())
    assert "provably changes the output" in flat
    assert "degrades honestly" in flat
    assert "same output regardless of input" in flat
