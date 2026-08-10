"""M20 exit criterion 1: a guard ships with the failure it rejects.

Fifteen defects on 2026-08-08, **eight of them one shape** — a gate that tests
something easier than it promises, and passes. Every one read as correct. Four of
the nine defects in the 2026-08-07 log were guards that could never fire, and
each had been read and approved.

`AGENTS.md` has recorded the countermeasure since then — *feed a guard a real bad
record before trusting it* — and three of the eight gates above were written
**after** it. Advice does not hold. So this file enumerates the capabilities that
report pass/fail and requires each to have a test proving it can say **no**.

The link is a marker, `@pytest.mark.rejects("<capability>")`, rather than a
naming convention or a grep for `passed is False`. A grep finds tests that
mention failure; a marker is a claim someone had to make on purpose, and it names
which gate the claim is about.

The bar the marker asserts is **red-then-green**: a test that passes with and
without the fix has proven nothing. That part cannot be automated, and the
marker's docstring is where the author says they did it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_CAPABILITIES = Path("src/labpilot/research_engine/execution/capabilities")
_TESTS = Path("tests")

#: Capabilities that have no rejection test yet, with the reason each is still
#: here. Every entry is a gate nobody has proven can fail — which is exactly the
#: state M20 exists to leave. The list only shrinks; adding to it fails the test
#: below that guards its length.
_UNPROVEN: dict[str, str] = {}


#: Capabilities whose every `passed=` is a literal `True` — they have no path to
#: a failing verdict at all. Sharper than "untested": a test cannot prove
#: rejection of something the code cannot do.
#:
#: Recorded rather than fixed here because the fix differs per capability and is
#: a behaviour change: either the capability has a real failure mode nobody
#: handles, or it has no verdict to give and reporting `passed=True` is a claim
#: of verification it never performed. Both are M20 work; neither is a rename.
_CANNOT_FAIL: dict[str, str] = {
    "reporting": ("4 return sites, all `passed=True` — writes a summary and calls it verified"),
    "runtime": (
        "2 return sites, both `passed=True` — provisions a runtime and cannot "
        "report that it did not"
    ),
    "stub": (
        "the no-op used until real capabilities register. Always passing is what "
        "it is *for*, and that is the point worth making: it is indistinguishable "
        "from a capability that verified something, and on 2026-08-08 four "
        "campaigns ran with codegen silently falling back. Listed rather than "
        "excused — a stub that reports `passed=True` is a gate that cannot fail "
        "wearing a capability's name."
    ),
}


def _capability_names() -> dict[str, Path]:
    """Capabilities that report a pass/fail verdict, by their registered name."""
    found: dict[str, Path] = {}
    for path in sorted(_CAPABILITIES.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "passed=" not in source:
            continue
        match = re.search(r'^\s{4}name\s*=\s*"([a-z_]+)"', source, re.M)
        if match:
            found[match.group(1)] = path
    return found


def _marked_capabilities() -> dict[str, list[str]]:
    """`@pytest.mark.rejects("x")` across the suite, by capability name."""
    marked: dict[str, list[str]] = {}
    for path in sorted(_TESTS.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                target = decorator.func
                if not (isinstance(target, ast.Attribute) and target.attr == "rejects"):
                    continue
                for argument in decorator.args:
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        marked.setdefault(argument.value, []).append(f"{path}::{node.name}")
    return marked


def test_every_capability_that_reports_a_verdict_can_be_shown_to_fail():
    """The criterion itself. A capability with no rejection test is a gate
    nobody has proven can say no — and on 2026-08-08 that described eight."""
    capabilities = _capability_names()
    marked = _marked_capabilities()

    assert capabilities, "no capabilities found — has the layout moved?"
    # `_CANNOT_FAIL` is excluded because no test can prove rejection of
    # something the code is unable to do — those are answered by
    # `test_no_capability_reports_a_verdict_it_cannot_withhold` instead.
    missing = sorted(set(capabilities) - set(marked) - set(_UNPROVEN) - set(_CANNOT_FAIL))

    assert not missing, (
        "these report pass/fail with no test proving they can reject a real bad "
        f'artifact: {missing}. Write one, marked `@pytest.mark.rejects("<name>")`, '
        "and confirm it is red before the fix and green after."
    )


def test_the_unproven_list_only_shrinks():
    """A list that grows once per incident is the curated-set pattern wearing
    verification's clothes — `15-gates-must-fail.md` names it as a trap. This
    one is allowed to exist only while it is being emptied."""
    assert len(_UNPROVEN) <= 0, (
        f"{len(_UNPROVEN)} capabilities are still unproven: {sorted(_UNPROVEN)}. "
        "Lower this number as they are covered; never raise it."
    )


def test_no_capability_reports_a_verdict_it_cannot_withhold():
    """The milestone's title, asked of the code rather than of its tests.

    A capability whose every `passed=` is a literal `True` is a gate that cannot
    fail — and unlike an untested gate, no test can fix it, because rejection is
    not something the code is able to do. Found on M20's first day, in two
    capabilities that had been running in every campaign.
    """
    unable = {}
    for name, path in _capability_names().items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        verdicts = [
            keyword.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg == "passed"
        ]
        if verdicts and all(
            isinstance(value, ast.Constant) and value.value is True for value in verdicts
        ):
            unable[name] = len(verdicts)

    unexpected = sorted(set(unable) - set(_CANNOT_FAIL))
    fixed = sorted(set(_CANNOT_FAIL) - set(unable))

    assert not unexpected, (
        f"these capabilities report a verdict they can never withhold: {unexpected}. "
        "Either give them a failing path or stop claiming they verified anything."
    )
    assert not fixed, f"these can fail now — remove them from _CANNOT_FAIL: {fixed}"


def test_a_marker_names_a_capability_that_exists():
    """A marker pointing at a renamed or deleted capability silently stops
    proving anything, which is the failure mode of the thing it enforces."""
    capabilities = _capability_names()

    orphans = {
        name: sites for name, sites in _marked_capabilities().items() if name not in capabilities
    }

    assert not orphans, f"`rejects` markers naming no capability: {orphans}"
