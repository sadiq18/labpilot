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


#: The trailing legend `tools_list()` always prints. Held here so the table
#: assertions can exclude it — it names all three statuses unconditionally,
#: so searching the raw output for "real"/"partial"/"fixed" finds them
#: whether or not a single tool carries that status. An earlier version of
#: this file did exactly that and asserted nothing.
_LEGEND_MARKER = "real = a declared input"


def _list_output() -> str:
    result = runner.invoke(cli_main.app, ["tools", "list"])
    assert result.exit_code == 0, result.stdout
    return result.stdout


def _table_only() -> str:
    """The rendered rows, with the legend cut off — whitespace-collapsed
    because Rich wraps long cells."""
    output = _list_output()
    head, sep, _legend = output.partition(_LEGEND_MARKER)
    assert sep, "legend marker not found — did the legend text change?"
    return " ".join(head.split())


def test_every_tool_appears_with_its_status() -> None:
    table = _table_only()
    for descriptor in default_tool_descriptors():
        assert descriptor.name in table, f"{descriptor.name} missing from `tools list`"


def test_each_status_is_rendered_in_the_table_not_only_the_legend() -> None:
    """Statuses must come from rows, not from the explanatory footer.

    Asserted against the table region only. The catalog currently carries all
    three, so all three must appear; if that ever stops being true the
    expectation below should change deliberately rather than keep passing on
    legend text.
    """
    table = _table_only()
    present = {descriptor.capability_status for descriptor in default_tool_descriptors()}
    assert present == {"real", "partial", "fixed"}, (
        f"catalog statuses changed to {sorted(present)} — update this test deliberately"
    )
    for status in sorted(present):
        assert status in table, f"status {status!r} is in the catalog but not rendered"


def test_varies_by_is_visible_for_a_real_tool() -> None:
    """The inventory must say *what* varies, not just that something does.

    `hypothesis_id` is `generate_plan`'s declared input; without this column
    an operator reads "real" and still cannot tell which knob to turn.
    """
    assert "hypothesis_id" in _table_only()


def test_the_legend_explains_the_three_statuses() -> None:
    flat = " ".join(_list_output().split())
    assert "provably changes the output" in flat
    assert "degrades honestly" in flat
    assert "same output regardless of input" in flat
